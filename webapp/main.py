import os
import json
import logging
import asyncio

from webapp.util import get_verbosity
from webapp.server import WebServer
from webapp.vision_rpc import VisionServer, VisionClient

log = logging.getLogger(__name__)

this_dir = os.path.dirname(os.path.realpath(__file__))


async def main():
    """Main entry point to camera node webserver"""
    # Load config file, with default in root of this repo
    config_file = os.getenv('CAMERA_CONFIG', os.path.join(this_dir, '../config.json'))
    with open(config_file, 'r') as f:
        config = json.load(f)
    # TODO: jsonschema validation

    # Set logging verbosity, with default at info level
    verbosity = config.get('verbosity', 2)
    logging.basicConfig(level=get_verbosity(verbosity), format='%(levelname)s : %(asctime)s : %(name)s : %(message)s')
    log.info(f'Read config file {config_file}')

    # Initialize camera/lidar RPC system
    log.info('Initializing camera')
    live_display = config.get('live_display', False)
    vision_server = VisionServer(config['camera'], live_display)
    vision_client = VisionClient(vision_server.req_q, vision_server.res_q)

    # Initialize webserver, feeding in the object that gives access to the vision system outputs
    web_server = WebServer(config['web'], vision_client)

    # Run main program loop
    log.info('Starting main loop')
    await asyncio.gather(
        vision_server.run(),
        web_server.run(),
        wait()
    )
    log.info('Stopping main loop')

    return


async def wait():
    while True:
        await asyncio.sleep(9999)


if __name__ == '__main__':
    asyncio.run(main())
