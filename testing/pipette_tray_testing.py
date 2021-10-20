import glob
import numpy as np
import cv2
import math

filenames = (glob.glob("testing_pipette/*.png"))
for indx, file in enumerate(filenames):
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([30, 50, 50]), np.array([50, 255, 255]))
    res = cv2.bitwise_and(img, img, mask=mask)
    res_color = res.copy()
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

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
        if rect_area > 10000:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # cv2.drawContours(img, [box], 0, (0,0,255))
            # cv2.putText(img, f'{int(rect_area)}', (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
            _rect = rect
            _box = box
    ret, thresh2 = cv2.threshold(res, 10, 255, cv2.THRESH_BINARY_INV)
    cnts2, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _rect2=[]
    for cnt in cnts2:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        check_point = cv2.pointPolygonTest(_box, rect[0], False)
        if rect[1][0] > 7 and rect[1][1] > 7 and check_point>-1:
            cv2.drawContours(img, [box], 0, (255, 255, 255))
            # cv2.putText(img, f'{int(rect[1][0]), int(rect[1][1])}', box[0], cv2.FONT_HERSHEY_PLAIN,1, (255,255,255), 1)
            _rect2.append(rect)

    tips = []
    pt_list = []
    spacing = 16.5
    for rect in _rect2:
        tol = 15

        x = rect[0][0]
        y = rect[0][1]
        w = rect[1][0]
        l = rect[1][1]
        angle = np.radians(rect[2])
        area = w*l
        if area> 26325:
            continue

        ranger = list(range(1,12))
        ranger = [ele*spacing for ele in ranger]
        width_check = False
        length_check = False
        for r in ranger:
            if r-tol<=w<=r+tol:
                width_check = True
                break
        for r in ranger:
            if r-tol<=l<=r+tol:
                length_check = True
                break
        if width_check and length_check:
            width_size = round(w/spacing)
            length_size = round(l/spacing)
            pt_array = []
            if width_size == length_size == 1:
                pt_array.append((x, y, 0))
            else:
                for ii in range(0,width_size):
                    for jj in range(0, length_size):
                        xn = x + spacing * (ii-width_size/2+.5) * math.cos(angle) - spacing * (jj-length_size/2+.5)*math.sin(angle)
                        yn = y + spacing * (ii-width_size/2+.5) * math.sin(angle) + spacing * (jj-length_size/2+.5)*math.cos(angle)
                        pt_array.append((xn, yn, angle))
            if pt_array:
                pt_list.append(pt_array)
    columns = []

    for pa in pt_list:
        skip = False
        group=[]
        if len(pa) > 1:
            angle = pa[0][2]
        else:
            angle = np.radians(_rect[2])
        for ii in columns:
            for jj in pa:
                exists = jj in ii
                if exists:
                    skip = True
                    break
        if skip:
            continue
        if angle > 1.39626:
            angle = angle -np.pi/2
        m = math.tan(angle)
        x = pa[0][0]
        y = pa[0][1]
        b = -m*x+y
        for i in pa:
            group.append(i)
        for pb in pt_list:
            if pb == pa:
                continue
            xx = pb[0][0]
            yy = pb[0][1]
            dist = abs(m*xx-yy+b)/math.sqrt(m*m+(-1*-1))
            if dist < 1:
                for i in pb:
                    group.append(i)
        if len(group) >1:
            columns.append(group)

    xvalues = []
    yvalues = []
    for sets in columns:
        for ii in sets:
            xvalues.append(ii[0])
            yvalues.append(ii[1])

    minx = min(xvalues)
    miny = min(yvalues)
    maxx = max(xvalues)
    maxy = max(yvalues)
    xc = (minx+maxx)/2
    yc = (miny+maxy)/2

    if maxx-minx>maxy-miny:
        num_rows = 12
        num_col = 8
    else:
        num_rows = 8
        num_col = 12
    target_array = []
    angle = np.angle(_rect[2])

    if angle > 1.3:
        angle = angle - np.pi / 2
    dewarp_spacing = 10
    angle = np.radians(angle)
    for ii in range(0, num_rows):
        for jj in range(0, num_col):
            xn = xc + dewarp_spacing*(ii-num_rows/2+.5) * math.cos(angle) - dewarp_spacing * (jj - num_col/2+.5)*math.sin(angle)
            yn = yc + dewarp_spacing*(ii-num_rows/2+.5)*math.sin(angle) + dewarp_spacing*(jj-num_col/2+.5)*math.cos(angle)
            target_array.append((xn,yn))


    cv2.imshow(str(indx), img)

cv2.waitKey(0)