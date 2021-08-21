import cv2
import numpy as np
def crop_rectangle(depth_image, rect):
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    #coordinate of the points in box points after the rectangle has been straightened
    dst_pts = np.array([[0, height-1],
                       [0, 0],
                       [width-1, 0],
                       [width-1, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped = cv2.warpPerspective(depth_image, M, (width, height))

    return warped