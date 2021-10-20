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
        self.workspace_height = None


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

    def establish_base(self):
        est_timer = self.config['initialization']['timer']
        thresh_timer = est_timer / 2
        store = {
            'center': [],
            'width': [],
            'height': []
        }
        for _ in range(est_timer):
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            depth_frame = self.temp_filter.process(depth_frame)
            depth = np.asanyarray(depth_frame.get_data())

            # identify tabletop, using histogram, highest bin expected to be target
            num_bins = self.config['initialization']['bins']
            mult = self.config['initialization']['mult']
            funk, binary, tbl_far = find_background(depth, num_bins, mult)

            if len(store['center']) < thresh_timer:
                center, width = centroid(binary)
                store['center'].append(center)
                store['width'].append(width)
                store['height'].append(tbl_far)
            else:
                center = np.mean(store['center'])
                width = np.mean(store['width'])
                height = np.mean(store['height'])

        self.workspace_center = center
        self.workspace_width = width
        self.workspace_height = height
        log.info(f'workspace height: {self.workspace_height}')
        log.info(f'depth scale: {self.depth_scale}')

    def depth_search(self, vis_on):
        try:
            while True:
                oi_config = self.config['object_identification']
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                color = np.asanyarray(color_frame.get_data())

                # get depth frame
                depth_frame = aligned_frames.get_depth_frame()

                # filter depth frame
                depth_frame = self.temp_filter.process(depth_frame)
                depth_filled_frame = self.hole_filter.process(depth_frame)
                depth = np.asanyarray(depth_frame.get_data())
                # depth_filled = np.asanyarray(depth_filled_frame.get_data())
                depthy = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)

                # find and isolate just the table (background)
                # funk, binary, tbl_far = find_background(depth, 2000, 6)
                funk = isolate_background(depth, self.workspace_height, 0, 200)
                funky = cv2.applyColorMap(cv2.convertScaleAbs(funk, alpha=0.03), cv2.COLORMAP_JET)
                junky = depthy - funky
                crop, l_bound, u_bound = crop_vert(junky, self.workspace_center, self.workspace_width)

                # contour detection
                im = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

                retval, threshold = cv2.threshold(im, oi_config['contour']['low'], oi_config['contour']['high'],
                                                  cv2.THRESH_BINARY_INV)
                im = cv2.cvtColor(threshold, cv2.COLOR_GRAY2RGB)
                contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > oi_config['area']['low'] and area < oi_config['area']['high']:
                        rect = cv2.minAreaRect(cnt)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        box_adj = np.array([[box[0][0], box[0][1] + l_bound],
                                            [box[1][0], box[1][1] + l_bound],
                                            [box[2][0], box[2][1] + l_bound],
                                            [box[3][0], box[3][1] + l_bound]])

                        display = cv2.drawContours(color, [box_adj], 0, (255, 255, 0), 2)
                if vis_on:
                    cv2.imshow('depth_feed', color)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        finally:
            self.pipeline.stop()

    def color_search(self, vis_on):
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                color = np.asanyarray(color_frame.get_data())
                color_to_bw = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)

                color_to_bw = cv2.bilateralFilter(color_to_bw, 11, 25, 25)
                color_to_bw = cv2.medianBlur(color_to_bw, 5)
                low_threshold = 20
                ratio = 5
                kernal = 21
                edges = cv2.Canny(color_to_bw, low_threshold, low_threshold * ratio, kernal)
                edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                # edges = cv2.bilateralFilter(edges, 9, 75, 75)
                edges = cv2.blur(edges, (3, 3))
                retval, threshold = cv2.threshold(edges, 5, 255,
                                                  cv2.THRESH_BINARY)
                threshold = cv2.cvtColor(threshold, cv2.COLOR_BGR2GRAY)
                contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > 1000 and area < 10000:
                        rect = cv2.minAreaRect(cnt)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        box_adj = np.array([[box[0][0], box[0][1]],
                                            [box[1][0], box[1][1]],
                                            [box[2][0], box[2][1]],
                                            [box[3][0], box[3][1]]])

                        display = cv2.drawContours(edges, [box_adj], 0, (255, 255, 0), 2)

                images = cv2.addWeighted(display, .8, edges, .2, 0.0)

                if vis_on:
                    cv2.imshow('color_feed', display)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            self.pipeline.stop()

    def run(self):
        """Main loop that continuously collects frames, analyzes them, and makes them available to the system."""
        try:
            while True:
                # Start timer for calculating frame rate
                tic = time.perf_counter()

                # Save timestamp for grabbed frame
                now = datetime.now(timezone.utc)

                # Grab frame
                oi_config = self.config['object_identification']
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                robot_frame = frames.get_depth_frame()

                depth_frame = self.temp_filter.process(depth_frame)
                depth_frame = self.hole_filter.process(depth_frame)
                prof = depth_frame.get_profile().as_video_stream_profile()
                depth_intrin = prof.get_intrinsics()

                prof = self.profile.get_stream(rs.stream.depth)
                robot_intrin = prof.as_video_stream_profile().get_intrinsics()

                depth = np.asanyarray(depth_frame.get_data())
                depthy = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)

                # find and isolate just the table (background)
                funk = isolate_background(depth, self.workspace_height, 10, 100)
                funky = cv2.applyColorMap(cv2.convertScaleAbs(funk, alpha=0.03), cv2.COLORMAP_JET)
                junky = depthy - funky
                crop, l_bound, u_bound = crop_vert(junky, self.workspace_center, self.workspace_width)

                im = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                retval, threshold = cv2.threshold(im, 100, 255, cv2.THRESH_BINARY_INV)
                kernel = np.ones((15, 15), np.uint8)
                # morph open erodes the outer edge, then dilates, removing small objects and merging nearby objects
                threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
                contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                color = np.asanyarray(color_frame.get_data())
                # Save raw data
                #   Passing through these data structures triggers:
                #   "RuntimeError: Error occured during execution of the processing block! See the log for more info"
                #   There's probably some buffer ownership mess that doesn't like this
                raw_data = RawData(now, np.copy(depth), np.copy(depthy), np.copy(color))
                self._write_raw(raw_data)

                # color detection
                low_threshold = 20
                color_to_bw = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
                color_to_bw = cv2.bilateralFilter(color_to_bw, 11, 25, 25)
                color_to_bw = cv2.medianBlur(color_to_bw, 5)
                ratio = 5
                kernal = 21
                edges = cv2.Canny(color_to_bw, low_threshold, low_threshold * ratio, kernal)
                edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                edges = cv2.blur(edges, (3, 3))
                retval_color, threshold_color = cv2.threshold(edges, 5, 255,
                                                              cv2.THRESH_BINARY)
                threshold_color = cv2.cvtColor(threshold_color, cv2.COLOR_BGR2GRAY)
                kernel = np.ones((21,21), np.uint8)
                threshold_color = cv2.morphologyEx(threshold_color, cv2.MORPH_CLOSE, kernel)
                contours_color, hierarchy_color = cv2.findContours(threshold_color, cv2.RETR_TREE,
                                                                   cv2.CHAIN_APPROX_SIMPLE)

                display = color

                # find the robot, robot uses the entire depth frame, which is larger than color, positions won't match
                robot_frame = self.hole_filter.process(robot_frame)
                robot_frame = self.temp_filter.process(robot_frame)
                robot = np.asanyarray(robot_frame.get_data())
                robot_data = robot.copy()
                # 2500 is roughly the height of the retracted spindle
                robot_data[robot_data > 1800] = 10000
                robot_im = cv2.applyColorMap(cv2.convertScaleAbs(robot_data, alpha=0.01), cv2.COLORMAP_JET)
                robot_im = cv2.cvtColor(robot_im, cv2.COLOR_BGR2GRAY)
                retval_robot, threshold_robot = cv2.threshold(robot_im, 50, 255,
                                                  cv2.THRESH_BINARY_INV)
                contours_robot, hierarchy_robot = cv2.findContours(threshold_robot, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                roboty = cv2.applyColorMap(cv2.convertScaleAbs(threshold_robot, alpha=0.03), cv2.COLORMAP_JET)

                # turn contours into objects
                robot_objects = self.check_robot(contours_robot, robot, robot_intrin)
                depth_objects, display = self.check_contours(contours, depth, depth_intrin, display, (255, 0, 255))
                color_objects, display = self.check_contours(contours_color, depth, depth_intrin, display, (255, 255, 255))

                robot_objects = [i for i in robot_objects if i]
                depth_objects = [i for i in depth_objects if i]
                color_objects = [i for i in color_objects if i]
                all_objects = robot_objects + depth_objects + color_objects
                self.objects = all_objects

                # Save analyzed data
                #   Passing through these data structures is fine since they're all created in our analysis code
                analyzed_data = AnalyzedData(now, all_objects)
                self._write_analyzed(analyzed_data)

                # Show detected objects (with which sensor detected what and heights) overlaid on color image
                if self.live_display:
                    cv2.imshow('depth_feed', display)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                toc = time.perf_counter()
                fps = 1 / (toc - tic)
                # log.debug(f'FPS: {int(round(fps))}')

        finally:
            self.pipeline.stop()

    def scale_point(self, depth_point):
        depth_point[0] = depth_point[0]*self.depth_scale
        depth_point[1] = depth_point[1]*self.depth_scale
        depth_point[2] = depth_point[2]*self.depth_scale
        return depth_point

    def check_robot(self, contours, depth, depth_intrin):
        objects = [[] for _ in contours]
        for cnt_index, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area > 3000 and area < 4000:
                circ = cv2.minEnclosingCircle(cnt)
                x = np.int0(circ[0][0])
                y = np.int0(circ[0][1])
                rad = circ[1]
                leg = np.int0(rad / math.sqrt(2) / 2)
                a = x - leg
                b = x + leg
                c = y - leg
                d = y + leg
                # log.info(f'leg: {a} {b} {c} {d}')
                crop_circ = depth[c:d, a:b]
                depth_mean, depth_std = self.isolate_mean(crop_circ)
                depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [circ[0][0], circ[0][1]], depth_mean)
                depth_point = self.scale_point(depth_point)
                height = -1 * (depth_point[2] - self.workspace_height) * self.depth_scale

                obj = {'type': 'robot', 'shape': 'circle', 'height': height,
                       'depth_mean': depth_mean,
                       'x_center': circ[0][0], 'y_center': circ[0][1],
                       'radius': rad, 'point': depth_point}
                objects[cnt_index] = obj
        return objects

    def check_contours(self, contours, depth, depth_intrin, display, color):
        oi_config = self.config['object_identification']
        objects = [[] for _ in contours]
        for cnt_index, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            display = cv2.drawContours(display, cnt, -1, color, 1)
            if oi_config['area']['low'] < area < oi_config['area']['high']:
                rect = cv2.minAreaRect(cnt)
                circ = cv2.minEnclosingCircle(cnt)
                area_rect = rect[1][0] * rect[1][1]
                area_circ = np.pi * np.square(circ[1])

                if area_rect > area_circ:
                    x = np.int0(circ[0][0])
                    y = np.int0(circ[0][1])
                    rad = np.int0(circ[1])
                    crop_circ = depth[x - rad:x + rad, y - rad:y + rad]
                    _crop_circ = crop_circ[crop_circ != 0]
                    depth_mean, depth_std = self.isolate_mean(_crop_circ)
                    depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth_mean)
                    depth_point = self.scale_point(depth_point)
                    height = -1 * (depth_point[2] - (self.workspace_height) * self.depth_scale)
                    if self.live_display:
                        display = cv2.circle(display, [x, y], rad, color, 2)
                        cv2.putText(display, f'{np.round(height, 3)}', (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                                    (36, 255, 12), 1)
                    obj = {'type': 'vision', 'shape': 'circle', 'height': height,
                           'depth_mean': depth_mean,
                           'x_center': circ[0][0], 'y_center': circ[0][1],
                           'radius': circ[1], 'point': depth_point}

                else:
                    if rect[1][0]*rect[1][1] > oi_config['area']['high']:
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
                        cv2.putText(display, f'{np.round(height, 3)}', (box[0][0], box[0][1]),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (36, 255, 12), 1)
                    obj = {'type': 'vision', 'shape': 'rect', 'height': height,
                           'depth_mean': depth_mean,
                           'x_center': rect[0][0], 'y_center': rect[0][1],
                           'width': rect[1][0], 'length': rect[1][1],
                           'angle': rect[2], 'point': depth_point}

                objects[cnt_index] = obj
        return objects, display