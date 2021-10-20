import pyrealsense2 as rs
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import asyncio
import cv2
import numpy as np
from rs_tools.pipe import find_background, centroid, crop_vert
import time

# this function is for testing camera tools and filtering
click_loc = 0
def crispy():
    pipeline = rs.pipeline()
    device_id = "f0221610"

    pc = rs.pointcloud()
    points = rs.points()
    # set camera config
    _config = rs.config()
    _config.enable_device(device_id)
    _config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
    _config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    _config.enable_record_to_file('test.bag')
    profile = pipeline.start(_config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.visual_preset, 5) #5 is short range
    depth_sensor.set_option(rs.option.confidence_threshold, 1) #3 i the highest
    depth_sensor.set_option(rs.option.noise_filtering, 6)
    align_to = rs.stream.depth
    align = rs.align(align_to)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    depth_scale = depth_scale*1000
    depth_scale = 1

    dec_filter = rs.decimation_filter()
    spat_filter = rs.spatial_filter()
    temp_filter = rs.temporal_filter()
    hole_filter = rs.hole_filling_filter()
    thresh_filter = rs.threshold_filter()
    temp_filter.set_option(rs.option.filter_smooth_alpha, .2)
    temp_filter.set_option(rs.option.filter_smooth_delta, 100)
    stor = {"center": [],
            "width": [],
            "depth": []}
    thresh_timer = 10

    x = np.linspace(1, 1024, 1024)
    y = np.sin(x)
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot()
    line1, = ax.plot(x, y)
    plt.ylim((1000, 10000))
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')

    clickstore = clickStore()
    cv2.namedWindow("feed")
    cv2.setMouseCallback('feed', clickstore.on_click)
    table_height = 0
    try:
        while True:
            # get aligned frames
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            depth = depth_frame

            # raw, at 1.27m from camera ~20mm of noise
            # depth = spat_filter.process(depth)
            depth = temp_filter.process(depth) #much tighter in the middle
            # depth = thresh_filter.process(depth)
            # depth = hole_filter.process(depth)
            # depth = dec_filter.process(depth) #changes size of the image, not necessarily desirable
            depth = np.asanyarray(depth.get_data())
            # point_cloud = pc.calculate(depth_frame)

            # todo : testing approach to background setting, currently the background is having abberations near edges, possibly from distortion
            funk, binary, tbl_far = find_background(depth, 2000, 6)
            crosssection = []
            clicker = clickstore.point
            if clickstore.orient == 'horiz':
                for i in depth[clicker]:
                    crosssection.append(float(i)*depth_scale)
                new_y = crosssection
                new_x = np.linspace(1, 1024, 1024)
            elif clickstore.orient == 'vert':
                for i in depth:
                    crosssection.append(float(i[clicker])*depth_scale)
                new_y = crosssection
                new_x = np.linspace(1, 768, 768)
            line1.set_xdata(new_x)
            line1.set_ydata(new_y)
            fig.canvas.draw()
            fig.canvas.flush_events()

            depthy = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
            funky = cv2.applyColorMap(cv2.convertScaleAbs(funk, alpha=0.03), cv2.COLORMAP_JET)
            junky = depthy - funky

            if len(stor['center']) < thresh_timer:
                center, width = centroid(binary)
                stor['center'].append(center)
                stor['width'].append(width)
                workspace_depth = width
                workspace_center = center
            else:
                workspace_center = np.mean(stor['center'])
                workspace_depth = np.mean(stor['width'])
                if table_height == 0:
                    table_height = tbl_far


                # plt.ylim((table_height*depth_scale+100*depth_scale, table_height*depth_scale-600*depth_scale))
                plt.ylim((table_height*depth_scale+100*depth_scale, 3500))

            crop, l_bound, u_bound = crop_vert(junky, workspace_center, workspace_depth)

            low_threshold = 50
            ratio = 2  # recommend ratio 2:1 - 3:1
            kernal = 3
            depth_image = np.asanyarray(junky)
            depth_to_jet = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            edges = cv2.Canny(depth_to_jet, low_threshold, low_threshold * ratio, kernal)
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            color = np.asanyarray(color_frame.get_data())
            images = cv2.addWeighted(depthy, .5, edges, .5, 0.0)
            # cv2.imshow('no filter', depthy)
            #
            if clickstore.orient == 'horiz':
                cv2.line(images, (0, clicker), (1024, clicker), (0, 255, 0), 2)
            elif clickstore.orient == 'vert':
                cv2.line(images, (clicker, 0), (clicker, 768), (0, 255, 0), 2)
            cv2.imshow('feed', images)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                # point_cloud = pc.calculate(depth_frame)
                # pc.map_to(color_frame)
    finally:
        pipeline.stop()

class clickStore:
    def __init__(self):
        self.point = 0
        self.orient = "horiz"
    def on_click(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            print('x:'+str(x)+' y:'+str(y))
            self.point = y
            self.orient = 'horiz'
        elif event == cv2.EVENT_RBUTTONDOWN:
            print('x:' + str(x) + ' y:' + str(y))
            self.point = x
            self.orient = 'vert'


if __name__ == '__main__':
    crispy()