import glob
import numpy as np
import cv2


filenames = (glob.glob("testing_pipette/*.png"))
for indx, file in enumerate(filenames):
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([30, 100, 50]), np.array([80, 255, 255]))
    res = cv2.bitwise_and(img, img, mask=mask)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    retval, thresh = cv2.threshold(res, 50, 255, cv2.THRESH_BINARY)
    cnts, hierarchy= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in cnts:
        rect = cv2.minAreaRect(cnt)
        x= rect[0][0]
        y = rect[0][1]
        rect_area = rect[1][0]*rect[1][1]
        if rect_area > 20000:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (0,0,255))
            # cv2.putText(img, f'{int(rect_area)}', (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)

    kernel = np.ones((3,3), np.uint8)
    erode = cv2.erode(gray, kernel, iterations=2)
    tophat = cv2.morphologyEx(erode, cv2.MORPH_TOPHAT, kernel)
    dilate = cv2.dilate(tophat, kernel, iterations=1)
    ret, thresh2 = cv2.threshold(dilate, 10, 255, cv2.THRESH_BINARY_INV)
    cnts2, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts2:
        rect = cv2.minAreaRect(cnt)
        x = rect[0][0]
        y = rect[0][1]
        rect_area = rect[1][0] * rect[1][1]
        if 80> rect_area > 30:
            cv2.circle(img, (int(x), int(y)), 0, (255, 255, 0), 3)

    cv2.imshow(str(indx), img)
cv2.waitKey(0)