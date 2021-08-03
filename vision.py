import logging
import cv2
import numpy as np
from rs_tools.pipe import setup_pipe, aligned, background, centroid, crop_horiz, crop_vert
log = logging.getLogger(__name__)

class Vision:
    def __init__(self, config):
        self.config = config
        self.device_id = config['device_id']

    async def connect(self):
        log.debug(f'Connecting to camera at {self.device_id}')
        self.pipeline, self.profile = setup_pipe(self.config, self.device_id)

    async def establish_base(self):
        est_timer = self.config['initialization']['timer']
        thresh_timer = est_timer/2
        stor = {"center": [],
                "width": [],
                "depth": []}
        try:
            while est_timer>0:
                frames = self.pipeline.wait_for_frames()
                images, depth, color, edges = aligned(self.config, frames)

                # identify tabletop, using histogram, highest bin expected to be target
                num_bins = self.config['initialization']['bins']
                mult = self.config['initialization']['mult']
                funk, binary, tbl_far = background(depth, num_bins, mult)

                if len(stor['center'])< thresh_timer:
                    center, width = centroid(binary)
                    stor['center'].append(center)
                    stor['width'].append(width)
                else:
                    center = np.mean(stor['center'])
                    width = np.mean(stor['width'])


                depthy = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
                funky = cv2.applyColorMap(cv2.convertScaleAbs(funk, alpha=0.03), cv2.COLORMAP_JET)

                crop, l_bound, u_bound = crop_vert(depthy - funky, center, width)

                # cv2.imshow('feed', crop)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                est_timer = est_timer - 1
        finally:
            self.workspace_center =center
            self.worksapce_depth =  width

    async def find_objs(self, action_q):
        try:
            while True:
                frames = self.pipeline.wait_for_frames()

                images, depth, color, edges = aligned(self.config, frames)
                depthy = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
                oi_config = self.config['object_identification']

                # remove the table {funky} from the depth data
                funk, binary, tbl_far = background(depth, oi_config['background']['bins'], oi_config['background']['mult'])
                funky = cv2.applyColorMap(cv2.convertScaleAbs(funk, alpha=0.03), cv2.COLORMAP_JET)
                junky = depthy-funky
                crop, l_bound, u_bound = crop_vert(junky, self.workspace_center, self.worksapce_depth)

                # contour detection
                im = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

                retval, threshold = cv2.threshold(im, oi_config['contour']['low'], oi_config['contour']['high'], cv2.THRESH_BINARY_INV)
                im = cv2.cvtColor(threshold, cv2.COLOR_GRAY2RGB)
                contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > oi_config['area']['low'] and area < oi_config['area']['high']:
                        rect = cv2.minAreaRect(cnt)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        box_adj = np.array([[box[0][0], box[0][1]+l_bound],
                                           [box[1][0], box[1][1]+l_bound],
                                           [box[2][0], box[2][1]+l_bound],
                                           [box[3][0], box[3][1]+l_bound]])

                        display = cv2.drawContours(images, [box_adj], 0, (255, 255, 0), 2)
                # cv2.imshow('align', display)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

        finally:
            self.pipeline.stop()
