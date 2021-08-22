import json
import aiohttp
import asyncio
import logging
from aiohttp import web

log = logging.getLogger(__name__)


class WebServer:
    def __init__(self, config):
        self.host = config.get('host', '0.0.0.0')  # default serve on all interfaces
        self.port = config['port']

    async def run(self):
        """Run web server that handles client connections"""

        api_app = web.Application()
        api_app.add_routes([
            web.get('/color-frame', self.handle_get_color_frame),
            web.get('/analyzed-frame', self.handle_get_analyzed_frame),
        ])

        app = web.Application()
        app.add_subapp('/api/v1/', api_app)

        # TODO: HTTPS/WSS
        # TODO: static site serving built webapp

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        log.info(f'Started web server on {self.host}:{self.port}')

    async def handle_get_color_frame(self, request):
        """Return the last captured color frame without analysis or annotations"""
        return web.Response(text="Hello, world")

    async def handle_get_analyzed_frame(self, request):
        """Return the last captured annotated depth + color frame as an annotated image with identified object data"""
        raise NotImplementedError
