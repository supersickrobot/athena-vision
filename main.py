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
    vision = Vision(config['camera'], config['safety'])
    await vision.establish_base()

    log.info('starting main loop')
    try:
        while True:
            await asyncio.gather(
                # server.run(),
                vision.search(action_q, True)
                # vision.depth_search(action_q, True),
                # vision.color_search(action_q, True)
                # vision.robot_safety(action_q, True)
            )
    except asyncio.CancelledError:
        log.info('Stopping main loop')




if __name__ == '__main__':
    asyncio.run(main())