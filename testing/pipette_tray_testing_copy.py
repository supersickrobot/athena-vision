import glob
import numpy as np
import cv2
import math

filenames = (glob.glob("testing_pipette/*.png"))
for indx, file in enumerate(filenames):
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([30, 100, 50]), np.array([80, 255, 255]))
    res = cv2.bitwise_and(img, img, mask=mask)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    new_img = cv2.erode(res, kernel, iterations=1)
    retval, thresh = cv2.threshold(res, 50, 255, cv2.THRESH_BINARY)
    cnts, hierarchy= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    new_img=thresh
    _rect = []
    _box = []
    for cnt in cnts:
        rect = cv2.minAreaRect(cnt)
        x = rect[0][0]
        y = rect[0][1]
        rect_area = rect[1][0]*rect[1][1]
        rect_shrunk = (rect[0], (rect[1][0]-20, rect[1][1]-20), rect[2])
        if rect_area > 10000:
            box = cv2.boxPoints(rect)
            box2 = cv2.boxPoints(rect_shrunk)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (0,0,255))
            # cv2.putText(img, f'{int(rect_area)}', (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
            _rect = rect
            _box = box2
    kernel = np.ones((3,3), np.uint8)
    erode = cv2.erode(gray, kernel, iterations=2)
    tophat = cv2.morphologyEx(erode, cv2.MORPH_TOPHAT, kernel)
    dilate = cv2.dilate(tophat, kernel, iterations=1)
    ret, thresh2 = cv2.threshold(dilate, 10, 255, cv2.THRESH_BINARY_INV)
    cnts2, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _targets = []
    for cnt in cnts2:
        rect = cv2.minAreaRect(cnt)
        x = rect[0][0]
        y = rect[0][1]
        check = cv2.pointPolygonTest(_box, (x,y), False)
        if check <=-1:
            continue
        rect_area = rect[1][0] * rect[1][1]
        if 80> rect_area > 30:
            cv2.circle(img, (int(x), int(y)), 0, (255, 255, 0), 3)
            _targets.append((x,y))

    if _rect[2] > 80:
        _rect = (_rect[0], (_rect[1][1], _rect[1][0]), _rect[2]-90)

    xc = _rect[0][0]
    yc = _rect[0][1]
    wc = _rect[0][0]/2
    lc = _rect[0][1]/2

    columns = []
    for tar in _targets:
        skip = False
        for i in columns:
            exists = tar in i
            if exists:
                skip = True
                break
        if skip:
            continue
        angle = _rect[2]
        m = math.tan(np.radians(angle))
        x = tar[0]
        y = tar[1]
        b = -m*x+y

        group = [tar]
        for tar2 in _targets:
            if tar == tar2:
                continue
            xx = tar2[0]
            yy = tar2[1]
            dist = abs(m*xx-yy+b)/math.sqrt(m*m+(-1*-1))
            if dist<1:
                group.append(tar2)

        if len(group)>1:
            columns.append(group)

    rows = []
    for tar in _targets:
        skip = False
        for i in rows:
            exists = tar in i
            if exists:
                skip = True
                break
        if skip:
            continue
        angle = _rect[2]-90
        m = math.tan(np.radians(angle))
        x = tar[0]
        y = tar[1]
        b = -m * x + y
        group = [tar]
        for tar2 in _targets:
            if tar == tar2:
                continue
            xx = tar2[0]
            yy = tar2[1]
            dist = abs(m * xx - yy + b) / math.sqrt(m * m + (-1 * -1))
            if dist < 1:
                group.append(tar2)

        if len(group) > 1:
            rows.append(group)

    for i in columns:
        m = math.tan(np.radians(_rect[2]))
        x = i[0][0]
        y = i[0][1]
        b = -m*x+y
        xvalues = []
        for j in i:
            xvalues.append(j[0])
        xmin = min(xvalues)
        xmax = max(xvalues)
        y1 = m*(xmin)+b
        y2 = m*(xmax)+b
        cv2.line(img, (int(xmin), int(y1)), (int(xmax), int(y2)), (255,255,255), 1)

    for i in rows:
        m = math.tan(np.radians(_rect[2]-90))
        x = i[0][0]
        y = i[0][1]
        b = -m * x + y
        yvalues = []
        for j in i:
            yvalues.append(j[1])
        ymin = min(yvalues)
        ymax = max(yvalues)
        if m >1000 or m < -1000:
            x1 = i[0][0]
            x2 = i[0][0]
        else:
            x1 = (ymin-b)/m
            x2 = (ymax-b)/m
        cv2.line(img, (int(x1), int(ymin)), (int(x2), int(ymax)), (255,255,255),1)

    distances_rows = []
    for i in rows:
        angle = _rect[2] - 90
        m = math.tan(np.radians(angle))
        x = i[0][0]
        y = i[0][1]
        b = -m * x + y
        dd = []
        for j in rows:
            x2 = j[0][0]
            y2 = j[0][1]
            b2 = -m*x2+y2
            distance = abs(b2-b)/math.sqrt(m*m+1)
            dd.append(distance)
        distances_rows.append(dd)


    cv2.imshow(str(indx), img)
    print(_rect)
cv2.waitKey(0)