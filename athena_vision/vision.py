import cv2
import time
import logging
import numpy as np
import pyrealsense2 as rs

from rs_tools.image_manipulation import crop_rectangle
from rs_tools.pipe import find_background, centroid, crop_vert, isolate_background, crop_below_workspace

log = logging.getLogger(__name__)


class Vision:
    """Realsense LIDAR driver and image processor

    This is designed to run in its own process because it is very computationally expensive. It saves its grabbed
        and analyzed frames in a local buffer that's made available to a VisionClient running in another process.
    """

    def __init__(self, config_camera, config_safety, live_display):
        self.config = config_camera
        self.safety = config_safety
        self.live_display = live_display

        self.device_id = self.config['device_id']
        self.safety_id = self.safety['device_id']
        self.objects = []
        self.pipeline = rs.pipeline()
        self.pipeline_safety = rs.pipeline()

        _config = rs.config()
        # _config.enable_device(self.device_id)
        rs.config.enable_device_from_file(_config, r'C:\Users\xpspectre\workspace\athena-vision\data\feed1.bag')
        _config.enable_stream(rs.stream.depth, self.config['depth']['resolution'][0],
                              self.config['depth']['resolution'][1], rs.format.z16, 30)
        # _config.enable_stream(rs.stream.color, self.config['color']['resolution'][0],
        #                       self.config['color']['resolution'][1], rs.format.bgr8, 30)
        # _config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
        _config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(_config)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        # depth_sensor.set_option(rs.option.visual_preset, 5)  # 5 is short range
        # depth_sensor.set_option(rs.option.confidence_threshold, 1)  # 3 i the highest
        # depth_sensor.set_option(rs.option.noise_filtering, 6)
        # depth_sensor.set_option(rs.option.inter_cam_sync_mode, 1)
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        self.depth_scale = depth_scale * 1000
        self.temp_filter = rs.temporal_filter()
        self.temp_filter.set_option(rs.option.filter_smooth_alpha, .2)
        self.temp_filter.set_option(rs.option.filter_smooth_delta, 100)
        self.hole_filter = rs.hole_filling_filter()

        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        self.depth_scale = self.depth_scale * 1000

        self.objects = []

    def isolate_height(self, cropped_image):
        depth_mean = np.mean(cropped_image)
        depth_std = np.std(cropped_image)
        height = -1 * (depth_mean - self.worksapce_height) * self.depth_scale
        return depth_mean, depth_std, height

    async def establish_base(self):
        est_timer = self.config['initialization']['timer']
        thresh_timer = est_timer / 2
        stor = {"center": [],
                "width": [],
                "height": []}
        try:
            while est_timer > 0:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                depth_frame = self.temp_filter.process(depth_frame)
                depth = np.asanyarray(depth_frame.get_data())

                # identify tabletop, using histogram, highest bin expected to be target
                num_bins = self.config['initialization']['bins']
                mult = self.config['initialization']['mult']
                funk, binary, tbl_far = find_background(depth, num_bins, mult)

                if len(stor['center']) < thresh_timer:
                    center, width = centroid(binary)
                    stor['center'].append(center)
                    stor['width'].append(width)
                    stor['height'].append(tbl_far)
                else:
                    center = np.mean(stor['center'])
                    width = np.mean(stor['width'])
                    height = np.mean(stor['height'])

                est_timer = est_timer - 1
        finally:
            self.workspace_center = center
            self.worksapce_width = width
            self.worksapce_height = height

    async def depth_search(self, action_q, vis_on):
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
                funk = isolate_background(depth, self.worksapce_height, 0, 200)
                funky = cv2.applyColorMap(cv2.convertScaleAbs(funk, alpha=0.03), cv2.COLORMAP_JET)
                junky = depthy - funky
                crop, l_bound, u_bound = crop_vert(junky, self.workspace_center, self.worksapce_width)

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

    async def color_search(self, action_q, vis_on):
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

    async def robot_safety(self, action_q, vis_on):
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)

                # get depth frame, search off table
                depth_frame = aligned_frames.get_depth_frame()
                depth = np.asanyarray(depth_frame.get_data())
                edge_loc = self.workspace_center + self.worksapce_width / 2
                crop = crop_below_workspace(depth, edge_loc)
                if vis_on:
                    cv2.imshow('safety_feed', crop)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            self.pipeline.stop()

    async def run(self):
        """Main loop that continuously collects frames, analyzes them, and makes them available to the system."""
        try:
            prev_process_time = 0
            new_process_time = 0
            while True:
                oi_config = self.config['object_identification']
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)

                # get frames
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                new_process_time = time.time()

                depth_frame = self.temp_filter.process(depth_frame)
                depth = np.asanyarray(depth_frame.get_data())
                depthy = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)

                # find and isolate just the table (background)
                # funk, binary, tbl_far = find_background(depth, 2000, 6)
                funk = isolate_background(depth, self.worksapce_height, 0, 200)
                funky = cv2.applyColorMap(cv2.convertScaleAbs(funk, alpha=0.03), cv2.COLORMAP_JET)
                junky = depthy - funky
                crop, l_bound, u_bound = crop_vert(junky, self.workspace_center, self.worksapce_width)

                # contour detection
                im = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

                retval, threshold = cv2.threshold(im, oi_config['contour']['low'], oi_config['contour']['high'],
                                                  cv2.THRESH_BINARY_INV)
                im = cv2.cvtColor(threshold, cv2.COLOR_GRAY2RGB)
                contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # color detection
                color = np.asanyarray(color_frame.get_data())
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
                contours_color, hierarchy_color = cv2.findContours(threshold_color, cv2.RETR_TREE,
                                                                   cv2.CHAIN_APPROX_SIMPLE)
                display = color
                depth_objects = [[] for xx in contours]
                color_objects = [[] for xx in contours_color]
                obj_bins = 100
                for cnt_index, cnt in enumerate(contours):
                    area = cv2.contourArea(cnt)
                    if area > oi_config['area']['low'] and area < oi_config['area']['high']:
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
                            depth_mean, depth_std, height = self.isolate_height(_crop_circ)
                            display = cv2.circle(display, [x, y], rad, (255, 255, 0), 2)
                            cv2.putText(display, f'{np.round(height, 3)}', (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                                        (36, 255, 12), 1)
                            obj = {'type': 'depth', 'shape': 'circle', 'dimensions': circ, 'height': height,
                                   'depth_mean': depth_mean, 'depth_std': depth_std}

                        else:
                            box = cv2.boxPoints(rect)
                            box = np.int0(box)
                            box_adj = np.array([[box[0][0], box[0][1] + l_bound],
                                                [box[1][0], box[1][1] + l_bound],
                                                [box[2][0], box[2][1] + l_bound],
                                                [box[3][0], box[3][1] + l_bound]])
                            crop_rect = crop_rectangle(depth, rect)
                            _crop_rect = crop_rect[crop_rect != 0]
                            depth_mean = np.mean(_crop_rect)
                            depth_std = np.std(_crop_rect)
                            hist, bins = np.histogram(_crop_rect, bins=obj_bins)
                            depth_val = bins[np.argmax(hist)]
                            height = -1 * (depth_val - self.worksapce_height) * self.depth_scale
                            depth_mean, depth_std, height = self.isolate_height(_crop_rect)
                            display = cv2.drawContours(display, [box_adj], 0, (255, 255, 0), 2)
                            cv2.putText(display, f'{np.round(height, 3)}', (box_adj[0][0], box_adj[0][1]),
                                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (36, 255, 12), 1)
                            obj = {'type': 'depth', 'shape': 'rect', 'dimensions': box, 'height': height,
                                   'depth_mean': depth_mean, 'depth_std': depth_std}
                        depth_objects[cnt_index] = obj

                for cnt_index, cnt in enumerate(contours_color):
                    area = cv2.contourArea(cnt)
                    if area > oi_config['area']['low'] and area < oi_config['area']['high']:
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
                            depth_mean, depth_mean, height = self.isolate_height(_crop_circ)
                            display = cv2.circle(display, [x, y], rad, (255, 0, 255), 2)
                            cv2.putText(display, f'{np.round(height, 3)}', (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                                        (36, 255, 12), 1)
                            obj = {'type': 'color', 'shape': 'circle', 'dimensions': circ, 'height': height,
                                   'depth_mean': depth_mean, 'depth_std': depth_std}

                        else:
                            box = cv2.boxPoints(rect)
                            box = np.int0(box)
                            crop_rect = crop_rectangle(depth, rect)
                            _crop_rect = crop_rect[crop_rect != 0]
                            depth_mean, depth_std, height = self.isolate_height(_crop_rect)
                            display = cv2.drawContours(display, [box], 0, (255, 0, 255), 2)
                            cv2.putText(display, f'{np.round(height, 3)}', (box[0][0], box[0][1]),
                                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (36, 255, 12), 1)
                            obj = {'type': 'depth', 'shape': 'rect', 'dimensions': box, 'height': height,
                                   'depth_mean': depth_mean, 'depth_std': depth_std}
                        color_objects[cnt_index] = obj

                depth_objects = [i for i in depth_objects if i]
                color_objects = [i for i in color_objects if i]
                all_objects = depth_objects + color_objects
                self.objects = all_objects
                if self.live_display:
                    cv2.imshow('depth_feed', display)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                fps = 1 / (new_process_time - prev_process_time)
                prev_process_time = new_process_time
                fps = int(fps)
                # print(fps)  # TODO: this was active
                # search depth
                # search color
                # clean/combine features
                # identification
        finally:
            self.pipeline.stop()


class VisionClient:
    """Client to the main Vision object that runs in the main program's event loop"""

    def __init__(self):
        raise NotImplementedError
