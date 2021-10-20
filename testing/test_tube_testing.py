import glob
import numpy as np
import cv2
import math

filenames = (glob.glob("testing_test_tubes_crop/*.png"))
for indx, file in enumerate(filenames):
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([110, 255, 255]))
    res = cv2.bitwise_and(img, img, mask=mask)
    res_color = res.copy()
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    retval, thresh = cv2.threshold(res, 50, 255, cv2.THRESH_BINARY)
    cnts, hierarchy= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    new_img=thresh
    _rect = []
    _box = []


    cv2.imshow(str(indx), res_color)

cv2.waitKey(0)