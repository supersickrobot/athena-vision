"""Remote procedure call (RPC)-like mechanism for using camera+image processing in another process"""
import io
import cv2
import asyncio
import logging
import threading
import traceback
import aioprocessing
import multiprocessing

from athena_vision.vision import Vision

log = logging.getLogger(__name__)


class VisionServer:
    """Server for the main Vision object"""

    def __init__(self, *args, **kwargs):
        """Instantiate main Vision object in another process. Takes the args that Vision does."""
        self._vision_args = args
        self._vision_kwargs = kwargs

        # Queues to communicate requests and responses with separate process
        self.req_q = aioprocessing.AioQueue()
        self.res_q = aioprocessing.AioQueue()

    async def run(self):
        """Wrapper that runs the server in the main program loop"""
        p = multiprocessing.Process(target=self._run_vision_process,
                                    args=(self.req_q, self.res_q, self._vision_args, self._vision_kwargs))
        p.start()

    def _run_vision_process(self, req_q, res_q, vision_args, vision_kwargs):
        """Vision runner in separate process. This is all synchronous threaded code."""
        # Set logging in separate process
        #   TODO: make verbosity configurable
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s : %(asctime)s : %(name)s : %(message)s')

        # Instantiate and initialize Vision object
        log.info('Running vision process')
        vision = Vision(*vision_args, **vision_kwargs)
        vision.establish_base()

        # Run the main Vision loop in another thread so the current one can handle RPCs
        t = threading.Thread(target=vision.run)
        t.start()

        while True:
            try:
                req = req_q.get()
                name, value = req
                log.debug(f'Got vision req: {req}')
                if name == 'color_img':
                    raw = vision.read_latest_raw()
                    _, buf = cv2.imencode('.jpg', raw.color)  # TODO: options like other file formats
                    res = io.BytesIO(buf)
                elif name == 'depth_img':
                    raw = vision.read_latest_raw()
                    _, buf = cv2.imencode('.jpg', raw.depth_img)
                    res = io.BytesIO(buf)
                elif name == 'analyzed_img':
                    analyzed = vision.read_latest_analyzed()
                    _, buf = cv2.imencode('.jpg', analyzed.annotated_img)
                    res = io.BytesIO(buf)
                elif name == 'analyzed_objects':
                    analyzed = vision.read_latest_analyzed()
                    res = analyzed.objects  # TODO: json-serilizable
                else:
                    log.error(f'Vision req: {req} not recognized. Ignoring.')
                res_q.put(res)
            except Exception as e:
                log.error(f'Error in vision processing: {e.args[0]}: {traceback.format_exc()}. Ignoring.')


class VisionClient:
    """Client to the Vision service object"""

    def __init__(self, req_q: aioprocessing.Queue, res_q: aioprocessing.Queue):
        self.req_q = req_q
        self.res_q = res_q

        # Make a lock that restricts the RPC system to 1 outstanding request at any time
        #   This is an asyncio lock that's only used in the program main loop
        self._lock = asyncio.Lock()

    async def _get(self, name):
        async with self._lock:
            self.req_q.put((name, None))
            return await self.res_q.coro_get()

    async def get_color_img(self):
        return await self._get('color_img')

    async def get_depth_img(self):
        return await self._get('depth_img')

    async def get_analyzed_img(self):
        return await self._get('analyzed_img')

    async def get_analyzed_objects(self):
        return await self._get('analyzed_objects')
