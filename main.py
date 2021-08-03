import asyncio
import logging
import os
import socket
import time
from server import Server
from config import configuration
from vision import Vision

log = logging.getLogger(__name__)
this_dir = os.path.dirname(os.path.realpath(__file__))

async def main():
    config = configuration()

    verbosity = config['verbosity']
    logging.basicConfig(level=verbosity, format='%(levelname)s : %(asctime)s : %(name)s : %(message)s')

    action_q = asyncio.Queue()
    display_q = asyncio.Queue()
    server = Server(action_q, display_q, config)
    log.info('initializing camera')
    vision = Vision(config['camera'])
    await vision.connect()
    center, width = await vision.establish_base()
    message = f'table found at center: {round(center)} w/ width {round(width)}'
    #  ,mp'], config['comm']['tcp_port']))

    log.info('starting main loop')
    try:
        while True:
            await asyncio.gather(
                server.run(),
                vision.find_objs(action_q),
            )
    except asyncio.CancelledError:
        log.info('Stopping main loop')



if __name__ == '__main__':
    asyncio.run(main())