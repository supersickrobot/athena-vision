import pyrealsense2 as rs
import cv2
import numpy as np
try:
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_device_from_file('feed4.bag')

    cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
    colorizer = rs.colorizer()
    pipeline.start(config)
    while True:
        # Get frameset of depth
        frames = pipeline.wait_for_frames()

        # Get depth frame
        depth_frame = frames.get_depth_frame()

        # Colorize depth frame to jet colormap
        depth_color_frame = colorizer.colorize(depth_frame)

        # Convert depth_frame to numpy array to render image in opencv
        depth_color_image = np.asanyarray(depth_color_frame.get_data())

        # Render image in opencv window
        cv2.imshow("Depth Stream", depth_color_image)
        key = cv2.waitKey(1)
        # if pressed escape exit program
        if key == 27:
            cv2.destroyAllWindows()
            break

finally:
    pass