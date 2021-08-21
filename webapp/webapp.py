import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from logging.handlers import QueueHandler
import webapp.read_and_respond
from webapp.websocket import WSHandler
from aiohttp import web
import asyncio
import json

log = logging.getLogger(__name__)

# load config here
async def main():
    # logging.basicConfig(level=logging.DEBUG, format='%(levelname)s : %(asctime)s : %(name)s : %(message)s')

    ws_handler = WSHandler()
    sub_app = web.Application()
    sub_app['ws'] = ws_handler

    ws_app = web.Application()
    ws_app.add_routes([web.get('/', ws_handler.handler)])

    app = web.Application()
    app.add_subapp('/api/v1', sub_app)
    app.add_subapp('/ws', ws_app)
    await asyncio.gather(
        # web._run_app(app, host=app_config_json["wsUrl"], port=app_config_json["wsUrlPort"]),  # using appConfig.json in root
        heartbeat(ws_handler),
        #this is your entry point for new async loops
    )

async def heartbeat(wshandler):

    while True:
        log.debug('start heartbeat')
        await asyncio.sleep(60)


if __name__ == '__main__':
    asyncio.run(main())