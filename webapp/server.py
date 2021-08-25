import logging
from aiohttp import web

log = logging.getLogger(__name__)


class WebServer:
    def __init__(self, config, vision_client):
        self.host = config.get('host', '0.0.0.0')  # default serve on all interfaces
        self.port = config['port']
        self.vision_client = vision_client

    async def run(self):
        """Run web server that handles client connections"""

        api_app = web.Application()
        api_app.add_routes([
            web.get('/color-img', self.handle_get_color_img),
            web.get('/depth-img', self.handle_get_depth_img),
            web.get('/analyzed-img', self.handle_get_analyzed_img),
            web.get('/analyzed-objects', self.handle_get_analyzed_objects),
            web.get('/ping', self.handle_ping)
        ])

        app = web.Application()
        app.add_subapp('/api/v1/', api_app)

        # TODO: HTTPS
        # TODO: static site serving built webapp

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        log.info(f'Started web server on {self.host}:{self.port}')

    async def handle_ping(self, request):
        print('ring-a-ding')
        ping = await self.vision_client.ping()
        return web.Response(body=ping, content_type='string')

    async def handle_get_color_img(self, request):
        """Return the last captured color image without analysis or annotations"""
        img = await self.vision_client.get_color_img()
        return web.Response(body=img, content_type='image/jpeg')

    async def handle_get_depth_img(self, request):
        """Return the last captured depth as color image without analysis or annotations"""
        img = await self.vision_client.get_depth_img()
        return web.Response(body=img, content_type='image/jpeg')

    async def handle_get_analyzed_img(self, request):
        """Return the last captured annotated depth + color as an annotated image with identified object data"""
        img = await self.vision_client.get_analyzed_img()
        return web.Response(body=img, content_type='image/jpeg')

    async def handle_get_analyzed_objects(self, request):
        """"Return the last captured identified objects list"""
        objects = await self.vision_client.get_analyzed_objects()
        return web.json_response(objects)

    # TODO: combined image+structured data responses
