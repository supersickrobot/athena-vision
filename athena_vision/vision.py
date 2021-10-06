import math
import flask
import cv2
import time
import logging
import threading
import collections
import numpy as np
import pyrealsense2 as rs
from datetime import datetime, timezone
from athena_vision.identify import identify_pipette_tray
from rs_tools.image_manipulation import crop_rectangle
from rs_tools.pipe import find_background, centroid, crop_vert, isolate_background

# Dev: turn numpy's runtime warnings into errors to debug
# import warnings
# warnings.simplefilter('error', RuntimeWarning)

log = logging.getLogger(__name__)

RawData = collections.namedtuple('RawData', ['time', 'depth', 'depth_img', 'color'])
AnalyzedData = collections.namedtuple('AnalyzedData', ['time', 'objects'])

class Vision:
    """Realsense LIDAR driver and image processor

    This is designed to run in its own process because it is very computationally expensive. It saves its grabbed
        and analyzed frames in a local buffer that's made available to a VisionClient running in another process.

    The functionality in here assumes it's running in its own thread/process. It doesn't use asyncio.
    """


    def __init__(self, config, live_display, buffer_capacity=32):
        self.config = config
        self.live_display = live_display
        self.buffer_capacity = buffer_capacity

        # TODO: validate config with jsonschema

        # Save collected data in ring buffers to make the latest (few) always available and restrict memory usage
        #   Control access via methods that are meant to be called from other threads
        self._raw_buffer = collections.deque(maxlen=buffer_capacity)
        self._raw_buffer_lock = threading.Lock()
        self._analyzed_buffer = collections.deque(maxlen=buffer_capacity)
        self._analyzed_buffer_lock = threading.Lock()
        self.objects = []

        # Connect to camera (or saved dataset to emulate camera)
        self.pipeline = rs.pipeline()

        cfg = rs.config()
        device_id = config.get('device_id')
        emulated_file = config.get('emulated_file')

        if emulated_file is None:
            cfg.enable_device(device_id)
        else:
            rs.config.enable_device_from_file(cfg, r'C:\Users\xpspectre\workspace\athena-vision\data\feed1.bag')

        cfg.enable_stream(rs.stream.depth, config['depth']['resolution'][0],
                          config['depth']['resolution'][1], rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, config['color']['resolution'][0],
                          config['color']['resolution'][1], rs.format.bgr8, 30)

        self.profile = self.pipeline.start(cfg)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        if emulated_file is None:
            # These are only settable on a real camera (they're readonly on an emulated camera)
            depth_sensor.set_option(rs.option.visual_preset, 5)  # 5 is short range
            depth_sensor.set_option(rs.option.confidence_threshold, 1)  # 3 i the highest
            depth_sensor.set_option(rs.option.noise_filtering, 6)
            depth_sensor.set_option(rs.option.inter_cam_sync_mode, 1)

        self.align = rs.align(rs.stream.color)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale() * 1000
        self.temp_filter = rs.temporal_filter()
        self.temp_filter.set_option(rs.option.filter_smooth_alpha, 0.1)
        self.temp_filter.set_option(rs.option.filter_smooth_delta, 100)
        self.hole_filter = rs.hole_filling_filter()

        self.workspace_center = None
        self.workspace_width = None
        self.workspace_height = config['table']['height']

    def _write_raw(self, raw):
        with self._raw_buffer_lock:
            self._raw_buffer.append(raw)

    def read_latest_raw(self):
        with self._raw_buffer_lock:
            return self._raw_buffer[-1]

    def _write_analyzed(self, analyzed):
        with self._analyzed_buffer_lock:
            self._analyzed_buffer.append(analyzed)

    def read_latest_analyzed(self):
        with self._analyzed_buffer_lock:
            return self._analyzed_buffer[-1]

    def isolate_mean(self, cropped_image):
        depth_mean = np.mean(cropped_image)
        depth_std = np.std(cropped_image)
        return depth_mean, depth_std

    def run(self):
        """Main loop that continuously collects frames, analyzes them, and makes them available to the system."""
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()

                depth_frame = self.temp_filter.process(depth_frame)
                # depth_frame = self.hole_filter.process(depth_frame)
                prof = depth_frame.get_profile().as_video_stream_profile()
                depth_intrin = prof.get_intrinsics()

                depth = np.asanyarray(depth_frame.get_data())
                depthy = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
                funk = isolate_background(depth, self.workspace_height, 20, 200)
                funky = cv2.applyColorMap(cv2.convertScaleAbs(funk, alpha=0.03), cv2.COLORMAP_JET)
                junky = depthy - funky

                isolate_table = cv2.cvtColor(junky, cv2.COLOR_BGR2GRAY)
                retval, threshold = cv2.threshold(isolate_table, 100, 255, cv2.THRESH_BINARY_INV)
                # retval, threshold = cv2.threshold(isolate_table, 200, 255, cv2.THRESH_BINARY_INV)
                kernel = np.ones((15, 15), np.uint8)
                # morph open erodes the outer edge, then dilates, removing small objects and merging nearby objects
                threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
                contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                color_frame = aligned_frames.get_color_frame()
                color = np.asanyarray(color_frame.get_data())
                display = color.copy()

                depth_objects, display = self.check_contours(contours, depth, depth_intrin, display, (255, 0, 255))
                depth_objects = [i for i in depth_objects if i]
                display = cv2.addWeighted(junky, 0.4, display, 0.6, 0)
                # 128 192

                for indx, obj in enumerate(depth_objects):
                    xc = obj['x_center']
                    yc = obj['y_center']
                    wc = obj['width']
                    lc = obj['length']
                    angle = obj['angle']
                    rect = ((xc, yc), (wc, lc), angle)

                    w = rect[1][0]
                    l = rect[1][1]
                    tol = 30
                    margin = 40
                    if 128 <= w <= 128 + tol and 190 <= l <= 190 + tol or \
                            128 <= l <= 128 + tol and 190 <= w <= 190 + tol:
                        crop, img_loc = self.create_crop(color, rect, margin)
                        cv2.imshow('pipette tray', crop)

                        response = identify_pipette_tray(crop)
                        if len(response) == 4:
                            angle_r, xr, yr, target_array = response
                        else:
                            continue
                        _target_array=[]
                        for i in target_array:
                            xa = i[0]
                            ya = i[1]
                            xn = img_loc[0][0] + xa #- img_loc[0][0]
                            yn = img_loc[1][0] + ya #- img_loc[0][1]
                            _target_array.append((xn, yn))
                        # cv2.putText(display, f'{int(img_loc[0][0], int(img_loc[0][1]))}', (int(xc), int(yc)), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
                        for ii in _target_array:
                            cv2.circle(display, (int(ii[0]), int(ii[1])), 0, (255, 255, 0), 5)
                        depth_objects[indx]['identity'] = 'pipette tray'
                        depth_objects[indx]['target'] = [angle_r, xr, yr]

                if self.live_display:
                    cv2.imshow('depth_feed', display)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        finally:
            self.pipeline.stop()

    def check_contours(self, contours, depth, depth_intrin, display, color):
        oi_config = self.config['object_identification']
        objects = [[] for _ in contours]
        for cnt_index, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            display = cv2.drawContours(display, cnt, -1, color, 1)
            if oi_config['area']['low'] < area < oi_config['area']['high']:
                rect = cv2.minAreaRect(cnt)
                area_rect = rect[1][0] * rect[1][1]

                if area_rect > oi_config['area']['high']:
                    continue
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                x = np.int0(rect[0][0])
                y = np.int0(rect[0][1])
                crop_rect = crop_rectangle(depth, rect)
                _crop_rect = crop_rect[crop_rect != 0]
                depth_mean, depth_std = self.isolate_mean(_crop_rect)
                depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth_mean)
                depth_point = self.scale_point(depth_point)
                height = -1 * (depth_point[2] - self.workspace_height* self.depth_scale)
                if self.live_display:
                    display = cv2.drawContours(display, [box], 0, color, 2)
                    cv2.putText(display, f'{int(rect[1][0]), int(rect[1][1])}', (box[0][0], box[0][1]),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (36, 255, 12), 1)
                obj = {'height': height,
                       'depth_mean': depth_mean,
                       'x_center': rect[0][0], 'y_center': rect[0][1],
                       'width': rect[1][0], 'length': rect[1][1],
                       'angle': rect[2], 'point': depth_point}

                objects[cnt_index] = obj
        return objects, display

    def scale_point(self, depth_point):
        depth_point[0] = depth_point[0]*self.depth_scale
        depth_point[1] = depth_point[1]*self.depth_scale
        depth_point[2] = depth_point[2]*self.depth_scale
        return depth_point

    def create_warp(self, img, rect, margin):
        w = rect[1][0]
        l = rect[1][1]

        new_rect = ((rect[0][0], rect[0][1]), (w + margin, l + margin), rect[2])
        box = cv2.boxPoints(new_rect)
        box = np.int0(box)
        wn = int(w + margin)
        ln = int(l + margin)
        src_pts = box.astype("float32")
        dst_pts = np.array([[0, ln],
                            [0, 0],
                            [wn, 0],
                            [wn, ln]], dtype='float32')
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(img, M, (wn, ln))
        return warped, M

    def create_crop(self, img, rect, margin):
        box = cv2.boxPoints(rect)
        xvalues =[]
        yvalues = []
        for i in box:
            xvalues.append(i[0])
            yvalues.append(i[1])

        xlow = int(min(xvalues)-margin/2)
        xhigh = int(max(xvalues)+margin/2)
        ylow = int(min(yvalues)-margin/2)
        yhigh = int(max(yvalues)+margin/2)
        if xlow < 0:
            xlow = 0
        if xhigh > 1920:
            xhigh = 1920
        if yhigh > 1080:
            yhigh = 1080
        if ylow < 0:
            ylow = 0
        crop = img[ylow:yhigh, xlow:xhigh]

        return crop, [[xlow, xhigh], [ylow, yhigh]]

    def project_warp_points(self, target_array, IM):
        _target_array = []
        for ii in target_array:
            xn = ii[0]
            yn = ii[1]
            xn, yn, z = np.dot(IM, [xn, yn] + [1])
            xn = xn/z
            yn = yn/z
            _target_array.append((xn,yn))
        return _target_array