import json
import asyncio
import logging
import traceback
import aiohttp
from aiohttp import web, WSCloseCode

from util import parse_url
from message import Action

log = logging.getLogger(__name__)


class Communicator:
    def __init__(self, config, action_q, display_q):
        # TODO: jsonschema validation
        self.config = config
        self.action_q = action_q
        self.display_q = display_q

        self.url = config['url']
        self.host, self.port = parse_url(self.url)
        # TODO: TLS

        # Aiohttp app runner
        app = web.Application()
        app.add_routes([web.get('/ws', self.handle_websocket)])
        self.runner = web.AppRunner(app)
        self.site = None

        # Set of connected websocket Clients
        self.clients = set()

    async def run(self):
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()

        log.info(f'Running comms on {self.url}')

        while True:
            try:
                display = await self.display_q.get()

                # Package into JSON string for websockets
                msg = json.dumps({'name': display.name, 'values': display.values, 'request_id': display.request_id})

                # Send to all clients
                for client in self.clients:
                    try:
                        client.send(msg)
                    except ClientError as e:
                        log.error(f'Client {e.args[0]}. Disconnecting.')
                        client.close()
                        self.clients.remove(client)

            except asyncio.CancelledError:
                log.info('Comms task cancelled.')
                for client in self.clients:
                    client.close()
                return
            except Exception as e:  # log other errors but keep going
                log.error(f'Exception in comms task: {e}\n{traceback.format_exc()}')

    async def handle_websocket(self, request):
        # Create and register new client
        client = await Client.new(request)
        self.clients.add(client)

        # Handle incoming messages from client
        try:
            async for msg in client.ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    # Unpack standard action message json
                    text = msg.data
                    log.debug(f'Action received: {text}')
                    action = json.loads(text)
                    name, values, request_id = action['name'], action.get('values'), action.get('request_id')

                    # Dispatch action
                    self.action_q.put_nowait(Action(name, values, request_id))

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    log.error(f'Client websocket connection closed with error: {client.ws.exception()}')

                else:
                    log.error(f'Client websocket unknown message type: {msg.type}')

        except asyncio.CancelledError:
            log.info('Client websocket handler task cancelled.')
        finally:
            self.clients.remove(client)

        log.info('Client websocket closed')
        return client.ws


class ClientError(Exception):
    pass


class Client:
    def __init__(self, ws):
        self.ws = ws  # used as the unique set element
        self.send_q = asyncio.Queue(50)  # per-client send queue with limit

    @classmethod
    async def new(cls, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        return Client(ws)

    def send(self, msg: str):
        """Enqueue a message to send immediately, failing if the client's send queue if full"""
        try:
            self.send_q.put_nowait(msg)
        except asyncio.QueueFull:
            raise ClientError('Send queue full')
        asyncio.create_task(self._send())

    async def _send(self):
        """Grab a message from own send queue and send it"""
        msg = self.send_q.get_nowait()
        await self.ws.send_str(msg)

    def __eq__(self, other):
        return self.ws == other.ws

    def __hash__(self):
        return hash(self.ws)

    def close(self):
        asyncio.create_task(self.ws.close(code=WSCloseCode.GOING_AWAY))
