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
                now = datetime.now(timezone.utc)
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()

                depth_frame = self.temp_filter.process(depth_frame)
                # depth_frame = self.hole_filter.process(depth_frame)
                prof = depth_frame.get_profile().as_video_stream_profile()
                depth_intrin = prof.get_intrinsics()

                depth = np.asanyarray(depth_frame.get_data())
                robot = depth.copy()
                robot[robot > 1900] = 10000
                robot[robot == 0] = 10000
                robot_im = cv2.applyColorMap(cv2.convertScaleAbs(robot, alpha=0.01), cv2.COLORMAP_JET)
                robot_im = cv2.cvtColor(robot_im, cv2.COLOR_BGR2GRAY)
                retval_robot, threshold_robot = cv2.threshold(robot_im, 50, 255,
                                                              cv2.THRESH_BINARY_INV)
                contours_robot, hierarchy_robot = cv2.findContours(threshold_robot, cv2.RETR_TREE,
                                                                   cv2.CHAIN_APPROX_SIMPLE)
                depthy = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
                funk = isolate_background(depth, self.workspace_height, 25, 200)
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
                cv2.drawContours(display, contours_robot, -1, (255,255,255), 1)
                contours = contours + contours_robot
                depth_objects, display = self.check_contours(contours, depth, depth_intrin, display, (255, 0, 255))
                depth_objects = [i for i in depth_objects if i]
                display = cv2.addWeighted(junky, 0.2, display, 0.8, 0)

                # search the depth objects for pipette trays and analyze them
                for indx, obj in enumerate(depth_objects):
                    xc = obj['x_center']
                    yc = obj['y_center']
                    wc = obj['width']
                    lc = obj['length']
                    angle = obj['angle']
                    rect = ((xc, yc), (wc, lc), angle)
                    cam_center_x = self.config['color']['resolution'][0] / 2
                    cam_center_y = self.config['color']['resolution'][1] / 2
                    tol = 30
                    if self.size_check(wc, lc, 50, 128, 190):
                        crop, [[xlow, xhigh], [ylow, yhigh]] = self.crop_mask(color, rect, 10)
                        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                        mask = cv2.inRange(hsv, np.array([30, 90, 100]), np.array([80, 255, 255]))
                        res = cv2.bitwise_and(crop, crop, mask=mask)
                        _res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
                        ret, thresh = cv2.threshold(_res, 10, 255, cv2.THRESH_BINARY)
                        cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        if cnts:
                            if len(cnts) > 1:
                                join_cnts = np.concatenate(cnts)
                            else:
                                join_cnts = np.asanyarray(cnts)
                                join_cnts = join_cnts[0]
                        rect = cv2.minAreaRect(join_cnts)
                        box = cv2.boxPoints(rect)
                        check_dist = 10000
                        iter1 = []
                        for iter, pt in enumerate(box):
                            xx = pt[0]
                            yy = pt[1]
                            dx = cam_center_x - (xx + xlow)
                            dy = cam_center_y - (yy + ylow)
                            dist = math.sqrt(dx * dx + dy * dy)
                            if check_dist > dist:
                                check_dist = dist
                                iter1 = iter
                        if iter1 == 3:
                            iter2 = 0
                        else:
                            iter2 = iter1 + 1

                        wr = rect[1][0]
                        lr = rect[1][1]
                        short_offset = 21
                        long_offset = 19
                        short_space = 13.25
                        long_space = 13.25
                        if wr > lr:
                            num_cols = 8
                            num_rows = 12
                            l_spacing = short_space
                            w_spacing = long_space
                            x_edge_offset = short_offset
                            y_edge_offset = long_offset
                        else:
                            num_rows = 8
                            num_cols = 12
                            l_spacing = long_space
                            w_spacing = short_space
                            x_edge_offset = long_offset
                            y_edge_offset = short_offset

                        if iter1 == 0:
                            orientx = -1
                            orienty = 1
                        elif iter1 == 1:
                            orientx = -1
                            orienty = -1
                        elif iter1 == 2:
                            orientx = 1
                            orienty = -1
                        else:
                            orientx = 1
                            orienty = 1

                        pt1 = box[iter1]
                        pt2 = box[iter2]
                        tube_array = []

                        box_angle = np.radians(rect[2])
                        for ii in range(0, num_rows):
                            for jj in range(0, num_cols):
                                xx = pt1[0] - (w_spacing * ii + x_edge_offset) * orientx * np.cos(box_angle) + (
                                            l_spacing * jj + y_edge_offset) * orienty * np.sin(box_angle)
                                yy = pt1[1] - (w_spacing * ii + x_edge_offset) * orientx * np.sin(box_angle) - (
                                            l_spacing * jj + y_edge_offset) * orienty * np.cos(box_angle)
                                xx = xx + xlow
                                yy = yy + ylow
                                tube_array.append((xx, yy))

                        for ii in tube_array:
                            cv2.circle(display, np.int0(ii), 5, (255, 255, 255))

                        cv2.line(res, np.int0(pt1), np.int0(pt2), (0, 255, 0), 2)
                        cv2.circle(res, np.int0(pt1), 5, (255, 255, 255))
                        cv2.putText(res, str(iter1), np.int0(pt1), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255))

                        cv2.drawContours(res, [np.int0(box)], 0, (0, 0, 255))
                        cv2.imshow('pipette', res)

                    elif self.size_check(wc, lc, 50, 160, 340): #check for regular test tube trays
                        crop, [[xlow, xhigh], [ylow, yhigh]] = self.crop_mask(color, rect, 10)
                        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                        mask = cv2.inRange(hsv, np.array([85, 125 , 100]), np.array([95, 255, 255]))
                        res = cv2.bitwise_and(crop, crop, mask=mask)
                        _res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
                        ret, thresh = cv2.threshold(_res, 10, 255, cv2.THRESH_BINARY)
                        cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        if cnts:
                            if len(cnts) > 1:
                                join_cnts = np.concatenate(cnts)
                            else:
                                join_cnts = np.asanyarray(cnts)
                                join_cnts = join_cnts[0]
                        rect = cv2.minAreaRect(join_cnts)
                        box = cv2.boxPoints(rect)
                        check_dist = 10000
                        iter1= []
                        for iter, pt in enumerate(box):
                            xx = pt[0]
                            yy = pt[1]
                            dx = cam_center_x - (xx + xlow)
                            dy = cam_center_y - (yy + ylow)
                            dist = math.sqrt(dx*dx + dy*dy)
                            if check_dist > dist:
                                check_dist = dist
                                iter1 = iter
                        if iter1 == 3:
                            iter2 = 0
                        else:
                            iter2 = iter1+1

                        wr = rect[1][0]
                        lr = rect[1][1]
                        short_offset = 20
                        long_offset = 18
                        short_space = 30.5
                        long_space = 32.5
                        if wr > lr:
                            num_cols = 5
                            num_rows = 10
                            l_spacing = short_space
                            w_spacing = long_space
                            x_edge_offset = short_offset
                            y_edge_offset = long_offset
                        else:
                            num_rows = 5
                            num_cols = 10
                            l_spacing = long_space
                            w_spacing = short_space
                            x_edge_offset = long_offset
                            y_edge_offset = short_offset

                        if iter1 == 0:
                            orientx = -1
                            orienty = 1
                        elif iter1 == 1:
                            orientx = -1
                            orienty = -1
                        elif iter1 == 2:
                            orientx = 1
                            orienty = -1
                        else:
                            orientx = 1
                            orienty = 1

                        pt1 = box[iter1]
                        pt2 = box[iter2]
                        tube_array = []

                        box_angle = np.radians(rect[2])
                        for ii in range(0,num_rows):
                            for jj in range(0, num_cols):
                                xx = pt1[0] - (w_spacing*ii+x_edge_offset)*orientx * np.cos(box_angle) + (l_spacing*jj+y_edge_offset)*orienty * np.sin(box_angle)
                                yy = pt1[1] - (w_spacing*ii+x_edge_offset)*orientx * np.sin(box_angle) - (l_spacing*jj+y_edge_offset)*orienty*np.cos(box_angle)
                                xx = xx + xlow
                                yy = yy + ylow
                                tube_array.append((xx,yy))

                        for ii in tube_array:
                            cv2.circle(display, np.int0(ii), 10, (255,255,255))


                        cv2.line(res, np.int0(pt1), np.int0(pt2), (0, 255, 0), 2)
                        cv2.circle(res, np.int0(pt1), 5, (255,255,255))
                        cv2.putText(res, str(iter1), np.int0(pt1), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255))

                        cv2.drawContours(res, [np.int0(box)],  0, (0,0,255))
                        cv2.imshow('tray', res)
                    elif self.size_check(wc, lc, tol, 115, 170):
                        height_low_mm = 5
                        height_high_mm = 15
                        proj_box, proj_rect = self.depth_trim(height_low_mm, height_high_mm, rect, depth)
                        if len(proj_box) != 1:
                            draw_box = np.int0(proj_box)
                            cv2.drawContours(display, [draw_box], 0, (255, 255, 255), 1)

                raw_data = RawData(now, np.copy(depth), np.copy(color), np.copy(display))
                self._write_raw(raw_data)
                all_objects = depth_objects
                analyzed_data = AnalyzedData(now, all_objects)
                self._write_analyzed(analyzed_data)
                resolution = color.size
                cam_center_x = int(self.config['color']['resolution'][0] / 2)
                cam_center_y = int(self.config['color']['resolution'][1] / 2)
                cv2.line(display, (1, cam_center_y), (cam_center_x*2, cam_center_y), (0, 0, 0), 1)
                cv2.line(display, (cam_center_x, 1), (cam_center_x, cam_center_y*2), (0, 0, 0), 1)
                if self.live_display:
                    cv2.imshow('depth_feed', display)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            self.pipeline.stop()


    def depth_trim(self, height_low_mm, height_high_mm, rect, depth):
        margin = 10
        height_low = self.mm_to_depth(self.workspace_height, height_low_mm)
        height_high = self.mm_to_depth(self.workspace_height, height_high_mm)
        crop, img_loc = self.crop_mask(depth, rect, margin)
        crop[crop > height_low] = 0
        crop[crop < height_high] = 0
        crop = cv2.applyColorMap(cv2.convertScaleAbs(crop, alpha=0.03), cv2.COLORMAP_JET)
        bw_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        retval, threshold = cv2.threshold(bw_crop, 50, 255, cv2.THRESH_BINARY)
        cnts, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        proj_box = [0]
        proj_rect = [0]
        if cnts:
            if len(cnts) > 1:
                join_cnts = np.concatenate(cnts)
            else:
                join_cnts = np.asanyarray(cnts)
                join_cnts = join_cnts[0]
            rect = cv2.minAreaRect(join_cnts)
            box = cv2.boxPoints(rect)
            proj_box = box
            proj_rect = rect
            for iter, pt in enumerate(proj_box):
                xn = img_loc[0][0] + pt[0]
                yn = img_loc[1][0] + pt[1]
                proj_box[iter] = (xn, yn)
                xc = rect[0][0] + img_loc[0][0]
                yc = rect[0][1] + img_loc[1][0]
                proj_rect = ((xc, yc), (rect[1][0], rect[1][1]), rect[2])
        return proj_box, proj_rect

    def size_check(self, wc, lc, tol, long_side, short_side):
        check = False
        if short_side <= wc <= short_side + tol and long_side <= lc <= long_side + tol or \
             short_side <= lc <= short_side + tol and long_side <= wc <= long_side + tol:
            check = True
        return check

    def mm_to_depth(self, table_height, height_in_mm):
        depth_value = table_height-4*height_in_mm
        return depth_value

    def depth_to_mm(self, table_height, depth_value):
        height_in_mm = (table_height-depth_value)/4
        return height_in_mm

    def check_contours(self, contours, depth, depth_intrin, display, color):
        oi_config = self.config['object_identification']
        objects = [[] for _ in contours]
        for cnt_index, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            # display = cv2.drawContours(display, cnt, -1, color, 1)
            if oi_config['area']['low'] < area < oi_config['area']['high']:
                rect = cv2.minAreaRect(cnt)
                circ = cv2.minEnclosingCircle(cnt)
                area_rect = rect[1][0] * rect[1][1]
                area_circ = circ[1]**2*np.pi
                if area_rect > oi_config['area']['high']:
                    continue
                if area_circ < area_rect:
                    xb = circ[0][0]
                    yb = circ[0][1]
                    db = circ[1]*2
                    rect = ((xb, yb), (db, db), 0)
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

    def crop_mask(self, image, rect, margin):
        img = image.copy()
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        img_shape = img.shape
        img_w = img_shape[0]
        img_l = img_shape[1]
        mask = np.zeros((img_w, img_l), np.uint8)
        cv2.drawContours(mask, [box], -1, (255), -1)

        color = (0,0,0)
        if img.dtype == 'uint16':
            img[mask != 255] = 0
        else:

            img[mask != 255] = color
        box = cv2.boxPoints(rect)
        xvalues = []
        yvalues = []
        for i in box:
            xvalues.append(i[0])
            yvalues.append(i[1])

        xlow = int(min(xvalues) - margin / 2)
        xhigh = int(max(xvalues) + margin / 2)
        ylow = int(min(yvalues) - margin / 2)
        yhigh = int(max(yvalues) + margin / 2)
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

    def create_crop(self, image, rect, margin):
        img = image.copy()
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