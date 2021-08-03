import os
import ssl
import json
import asyncio
import aiohttp
from aiohttp import web, WSCloseCode
import logging
import traceback
from message import Action

log = logging.getLogger(__name__)

class Server:
    def __init__(self, action_q, display_q, config):
        self.action_queue = action_q
        self.display_queue = display_q

        self.config = config
        web_config = config['web']

        self.host = web_config['host']
        self.port = web_config['port']

        app = web.Application()
        app.add_routes([web.get('/ws', self.websocket_handler)])

        app.on_shutdown.append(self.on_shutdown)

    async def run(self):
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port, ssl_context=self.ssl_context)
        await self.site.start()
        log.info(f'Running websocket on {self.url}')

        log.info('Running display queue sender')
        while True:
            try:
                display = await self.display_queue.get()

                # Pack into string/JSON form for ws
                msg = json.dumps({
                    'name': display.name,
                    'values': display.values
                })

                for ws in self.clients:
                    # log.debug(f'Sending Display JSON to client: {msg}')  # TODO: Client name
                    await ws.send_str(msg)

            except asyncio.CancelledError:
                log.info('Detected task cancellation. Stopping display queue sender.')
                return

            except Exception as e:  # Handle any other exception in actions and let the program keep running
                log.error(f'Exception occurred in display handler: {e}\n{traceback.format_exc()}')

    async def root_handler(self, request):
        """Handle browser going to root to get index.html main page"""
        return aiohttp.web.HTTPFound('/index.html')

    async def websocket_handler(self, request):
        """Handle websocket connection with 1 client"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        # Register client
        log.info('New client registered')
        self.clients.add(ws)

        # Handle further messages in a loop
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    name, values = recv_action(msg)

                    # Handle client 1st message: send config to client and dispatch action to send current state
                    if name == 'config':
                        await ws.send_json({'name': 'config', 'values': self.config})
                        await self.action_queue.put(Action('dump'))
                        continue

                    await self.action_queue.put(Action(name, values))

                    # For now, close the server on the 1 remote client close
                    if name == 'close':
                        break

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    log.error(f'Websocket connection closed with exception: {ws.exception()}')

                else:
                    log.error(f'Unknown websocket message type')

        except asyncio.CancelledError:
            log.info('Detected task cancellation. Stopping websocket handler.')

        finally:
            self.clients.remove(ws)

        log.info(f'Websocket connection closed')
        return ws

    async def on_shutdown(self, app):
        for ws in self.clients:
            await ws.close(code=WSCloseCode.GOING_AWAY, message='Server shutdown')

    async def close(self):
        await self.runner.cleanup()

def recv_action(msg):
    text = msg.data
    log.debug(f'Action received {text}')  # this is pretty verbose
    action = json.loads(text)
    return action['name'], action['values']

