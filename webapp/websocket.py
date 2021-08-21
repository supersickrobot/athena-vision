from aiohttp import web
import aiohttp
import logging
import json
from webapp import read_and_respond as rr
log = logging.getLogger(__name__)

class WSHandler:
    def __init__(self):
        self.clients = set()

    async def handler(self, request):
        client = web.WebSocketResponse()
        await client.prepare(request)

        # Register clients
        #   Simple scheme where the client object is stored directly
        self.clients.add(client)

        async for msg in client:
            if msg.type == aiohttp.WSMsgType.TEXT:
                body = json.loads(msg.data)

                # Handle disconnect
                if body['name'] == 'close':
                    log.info(f'Disconnect from client')
                    await client.close()
                    self.clients.remove(client)
                    break

                await rr.read_respond(self, body)

            elif msg.type == aiohttp.WSMsgType.ERROR:
                log.error(f'Websocket connection closed with exception: {client.exception()}')

    async def send(self, name, values):
        for client in self.clients:
            if client.closed:
                # clean up closed sockets
                await client.close()
                continue
            try:
                await client.send_str(json.dumps({'name': name, 'values': values}))
            except Exception as e:
                log.error(f'error during send:{e}')