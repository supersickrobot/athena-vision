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
from rs_tools.pipe import isolate_background

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


    def __init__(self, config, id_config, live_display, buffer_capacity=32):
        self.config = config
        self.id_config = id_config
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
                color_frame = aligned_frames.get_color_frame()
                color = np.asanyarray(color_frame.get_data())
                display = color.copy()

                depth_frame = self.temp_filter.process(depth_frame)
                # depth_frame = self.hole_filter.process(depth_frame)
                prof = depth_frame.get_profile().as_video_stream_profile()
                depth_intrin = prof.get_intrinsics()

                depth = np.asanyarray(depth_frame.get_data())
                _depth = depth.copy()
                _depth[_depth < 3700] = 0
                depthy = cv2.applyColorMap(cv2.convertScaleAbs(_depth, alpha=0.03), cv2.COLORMAP_JET)
                funk = isolate_background(_depth, self.workspace_height, 30, 200)
                funky = cv2.applyColorMap(cv2.convertScaleAbs(funk, alpha=0.03), cv2.COLORMAP_JET)
                junky = depthy - funky

                isolate_table = cv2.cvtColor(junky, cv2.COLOR_BGR2GRAY)
                retval, threshold = cv2.threshold(isolate_table, 100, 255, cv2.THRESH_BINARY)
                # retval, threshold = cv2.threshold(isolate_table, 200, 255, cv2.THRESH_BINARY_INV)
                kernel = np.ones((15, 15), np.uint8)
                # morph open erodes the outer edge, then dilates, removing small objects and merging nearby objects
                threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
                contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                depth_objects, display = self.check_contours(contours, depth, depth_intrin, display, (255, 0, 255))
                depth_objects = [i for i in depth_objects if i]
                display = cv2.addWeighted(junky, 0.2, display, 0.8, 0)

                # search the depth objects for pipette trays and analyze them
                for indx, obj in enumerate(depth_objects):
                    rect = obj['rect']
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    if self.size_check_from_3D_data(self.id_config['pipette tip tray 1mL'], obj, 10, depth_intrin):
                        pipette_center = self.edge_search(self.id_config['pipette tip tray 1mL'], color, rect,
                                                          depth_intrin, display, False)
                        if pipette_center == []:
                            pipette_center = self.edge_search(self.id_config['pipette tip tray 50uL'], color, rect,
                                                              depth_intrin, display, False)
                            if pipette_center == []:
                                continue
                            else:
                                depth_objects[indx]['identity'] = 'pipette tip tray 50uL'
                                # cv2.putText(display, f'pipette 50uL',
                                #             (box[1][0], box[1][1]),
                                #             cv2.FONT_HERSHEY_DUPLEX, 0.5, (36, 255, 12), 1)
                        else:
                            depth_objects[indx]['identity'] = 'pipette tip tray 1mL'

                        depth_objects[indx]['identified center'] = pipette_center

                    elif self.size_check_from_3D_data(self.id_config['test tube tray'], obj, 10, depth_intrin):
                        tray_center = self.edge_search(self.id_config['test tube tray'], color, rect, depth_intrin,
                                                       display, False)
                        if tray_center == []:
                            continue
                        depth_objects[indx]['identified center'] = tray_center
                        depth_objects[indx]['identity'] = 'test tube tray'

                    elif self.size_check_from_3D_data(self.id_config['waste bin'], obj, 10, depth_intrin):
                        depth_objects[indx]['identified center'] = obj['point']
                        depth_objects[indx]['identity'] = 'waste bin'

                raw_data = RawData(now, np.copy(depth), np.copy(color), np.copy(display))
                self._write_raw(raw_data)
                all_objects = depth_objects
                analyzed_data = AnalyzedData(now, all_objects)
                self._write_analyzed(analyzed_data)
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


    def edge_search(self, config, image, rect_o, depth_intrin, display, show):
        search_color_low = config['search_color_low']
        search_color_high = config['search_color_high']
        width_offset = config['width offset']
        length_offset = config['length offset']
        width_space = config['width spacing']
        length_space = config['length spacing']
        num_of_slots_width = config['num of slots width']
        num_of_slots_length = config['num of slots length']
        height_mm = config['height in mm']
        height_relative_to_robot = self.mm_to_camera_height(height_mm)
        circle_size = config['circle size in pixels']
        color_thresh = config['color threshold']
        height = self.mm_to_depth(height_mm)
        # todo the crop slows the image processing way donw, figure out a way to do this by blacking out the original image without cropping
        crop, [[xlow, xhigh], [ylow, yhigh]] = self.crop_mask(image, rect_o, 10)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(search_color_low), np.array(search_color_high))
        res = cv2.bitwise_and(crop, crop, mask=mask)
        _res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(_res, 10, 255, cv2.THRESH_BINARY)
        if show:
            cv2.imshow('crop', thresh)
        if sum(sum(_res)) < color_thresh or sum(sum(_res) > color_thresh + 5000):
            return []
        cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        new_cnts = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area > 20:
                new_cnts.append(cnt)
        if new_cnts:
            if len(new_cnts) > 1:
                join_cnts = np.concatenate(new_cnts)
            else:
                join_cnts = np.asanyarray(new_cnts)
                join_cnts = join_cnts[0]
        else:
            return
        rect = cv2.minAreaRect(join_cnts)
        box = cv2.boxPoints(rect)
        check_dist = 10000
        iter1 = []
        cam_center_x = self.config['color']['resolution'][0] / 2
        cam_center_y = self.config['color']['resolution'][1] / 2
        for iter, pt in enumerate(box):
            xx = pt[0]
            yy = pt[1]
            dx = cam_center_x - (xx + xlow)
            dy = cam_center_y - (yy + ylow)
            dist = math.sqrt(dx * dx + dy * dy)
            if check_dist > dist:
                check_dist = dist
                iter1 = iter

        y_center_offset = width_offset+(num_of_slots_width-1)*width_space/2
        x_center_offset = length_offset+(num_of_slots_length-1)*length_space/2

        box_angle = np.radians(rect_o[2])
        orientx = 1
        orienty = 1
        if np.degrees(box_angle) >= 90 and iter1 ==3:
            orienty = orienty * -1
        if np.degrees(box_angle) <= 90 and iter1 == 0:
            orientx = orientx * -1
        if iter1 ==2:
            orienty = orienty * -1
        if iter1 == 2 and np.degrees(box_angle) >= 90:
            orientx = orientx * -1

        pt1 = box[iter1]

        corner_pos = rs.rs2_deproject_pixel_to_point(depth_intrin, [pt1[0]+xlow, pt1[1]+ylow], height)
        corner_pos = self.scale_point(corner_pos)
        new_x = corner_pos[0] - x_center_offset * orientx*np.cos(box_angle) + y_center_offset * orienty * np.sin(box_angle)
        new_y = corner_pos[1] - x_center_offset*orientx*np.sin(box_angle) - y_center_offset*orienty*np.cos(box_angle)
        center_pos = [new_x, new_y, corner_pos[2]]
        px_cent = rs.rs2_project_point_to_pixel(depth_intrin, center_pos)
        tube_array = []
        for ii in range(0, num_of_slots_length):
            for jj in range(0, num_of_slots_width):
                xx = corner_pos[0] - (length_space * ii + length_offset) * orientx * np.cos(box_angle) + (
                    width_space * jj + width_offset) * orienty * np.sin(box_angle)
                yy = corner_pos[1] - (length_space * ii + length_offset) * orientx * np.sin(box_angle) - (
                    width_space * jj + width_offset) * orienty * np.cos(box_angle)
                tube_array.append((xx, yy, corner_pos[2]))

        cv2.circle(display, np.int0((pt1[0] + xlow, pt1[1] + ylow)), 2, (255, 255, 0), 1)
        pixel_corner = rs.rs2_project_point_to_pixel(depth_intrin, corner_pos)
        cv2.circle(display, np.int0(pixel_corner), 5, (255, 0, 255), 1)
        cv2.circle(display, np.int0(px_cent), 5, (255, 0, 255), 1)
        for indx, ii in enumerate(tube_array):
            pixel_pt = rs.rs2_project_point_to_pixel(depth_intrin, ii)
            cv2.circle(display, np.int0(pixel_pt), circle_size, (255, 255, 255))
            # cv2.putText(display, f'{int(indx)}', np.int0(pixel_pt), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        return center_pos

    def find_tip(self, color, slot, px_diam, angle, camera_angle):

            if camera_angle < 10:
                mask_lower = np.array([40, 5, 110])
                mask_upper = np.array([70, 60, 255])
            else:
                mask_lower = np.array([20, 50, 0])
                mask_upper = np.array([140, 255, 20])
            slotx = slot[0]
            sloty = slot[1]
            rect = ((slotx, sloty), (px_diam, px_diam), np.degrees(angle))

            img_copy = color.copy()
            crop, _  = self.crop_mask(img_copy, rect, 5)

            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, mask_lower, mask_upper)
            res = cv2.bitwise_and(crop, crop, mask=mask)
            _res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
            sum_of_pixels = sum(sum(_res))
            width = int(crop.shape[1] * 10)
            height = int(crop.shape[0] * 10)
            dim = (width, height)
            if sum_of_pixels > 400:
                return True
            else:
                return False
    def size_check(self, wc, lc, tol, long_side, short_side):
        check = False
        if short_side <= wc <= short_side + tol and long_side <= lc <= long_side + tol or \
             short_side <= lc <= short_side + tol and long_side <= wc <= long_side + tol:
            check = True
        return check

    def size_check_from_3D_data(self, identity_dict, obj, tol, depth_intrin):
        check = False
        rect = obj['rect']
        width_in_mm = identity_dict['width in mm']
        length_in_mm = identity_dict['length in mm']
        height_in_mm = identity_dict['height in mm']
        box = cv2.boxPoints(rect)
        c_3d = []
        for corner in box:
            depth_height = self.mm_to_camera_height(height_in_mm)
            corner_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, corner, depth_height)
            c_3d.append(corner_3d)
        depth_pt = obj['point']
        cam_ang = np.arcsin(np.sqrt(depth_pt[0] ** 2 + depth_pt[1] ** 2) / depth_pt[2])
        offset_by_height = np.tan(cam_ang)*height_in_mm
        dx = c_3d[0][0]-c_3d[1][0]
        dy = c_3d[0][1]-c_3d[1][1]
        check_width = np.sqrt(dx**2+dy**2)

        dx = c_3d[1][0]-c_3d[2][0]
        dy = c_3d[1][1]-c_3d[2][1]
        check_length = np.sqrt(dx**2+dy**2)

        if width_in_mm-tol <= check_width <= width_in_mm+tol+offset_by_height and\
            length_in_mm - tol <= check_length < length_in_mm + tol + offset_by_height:
            check = True
        return check

    def mm_to_camera_height(self, height_in_mm):
        table_height = self.workspace_height/4
        z_height = table_height-height_in_mm
        return z_height

    def mm_to_depth(self, height_in_mm):
        table_height = self.workspace_height
        depth_value = table_height-4*height_in_mm
        return depth_value

    def depth_to_mm(self, depth_value):
        table_height = self.workspace_height
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
                l, w = rect[1]
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
                if l < w:
                    rect = (rect[0], (w, l), rect[2]+90)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                x = np.int0(rect[0][0])
                y = np.int0(rect[0][1])
                crop_rect = crop_rectangle(depth, rect)
                _crop_rect = crop_rect[crop_rect != 0]
                depth_mean, depth_std = self.isolate_mean(_crop_rect)
                if np.isnan(depth_mean):
                    continue
                depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth_mean)
                depth_point = self.scale_point(depth_point)
                height = -1 * (depth_point[2] - self.workspace_height * self.depth_scale)
                if self.live_display and depth_mean > 2000:
                    display = cv2.drawContours(display, [box], 0, color, 2)
                    cv2.putText(display, f'{int(depth_point[0]), int(depth_point[1])}', (box[0][0], box[0][1]),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (36, 255, 12), 1)
                obj = {'height': height,
                       'depth_mean': depth_mean, 'rect': rect, 'angle': np.radians(rect[2]), 'point': depth_point}

                objects[cnt_index] = obj
        return objects, display

    def scale_point(self, depth_point):
        depth_point[0] = depth_point[0]*self.depth_scale
        depth_point[1] = depth_point[1]*self.depth_scale
        depth_point[2] = depth_point[2]*self.depth_scale
        return depth_point

    def crop_mask(self, image, rect, margin):
        img_copy = image.copy()
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
        crop = img_copy[ylow:yhigh, xlow:xhigh]

        _rect = ((rect[0][0]-xlow, rect[0][1]-ylow), (rect[1][0]+margin, rect[1][1]+margin), rect[2])
        box = cv2.boxPoints(_rect)
        _box = np.int0(box)
        crop_shape = crop.shape
        img_w = crop_shape[0]
        img_l = crop_shape[1]
        mask = np.zeros((img_w, img_l), np.uint8)
        cv2.drawContours(mask, [_box], -1, (255), -1)

        color = (0,0,0)
        if crop.dtype == 'uint16':
            crop[mask != 255] = 0
        else:
            crop[mask != 255] = color

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