import asyncio
import logging
import os
import socket
import time

from config import configuration
from vision import Vision
from http.server import BaseHTTPRequestHandler, HTTPServer

log = logging.getLogger(__name__)
this_dir = os.path.dirname(os.path.realpath(__file__))

async def main():
    config = configuration()

    verbosity = config['verbosity']
    logging.basicConfig(level=verbosity, format='%(levelname)s : %(asctime)s : %(name)s : %(message)s')

    log.info('initializing camera')
    vision = Vision(config['camera'], config['safety'])
    await vision.establish_base()

    log.info('starting main loop')
    try:
        while True:
            await asyncio.gather(
                vision.search(False)
            )
    except asyncio.CancelledError:
        log.info('Stopping main loop')


if __name__ == '__main__':
    asyncio.run(main())