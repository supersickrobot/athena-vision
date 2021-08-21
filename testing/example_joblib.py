def check_depth_contours(self, cnt):
    oi_config = self.config['object_identification']

    area = cv2.contourArea(cnt)
    if area > oi_config['area']['low'] and area < oi_config['area']['high']:
        dep_itr = [[] for x in range(oi_config['area']['high'] + 1)]
        rect = cv2.minAreaRect(cnt)
        circ = cv2.minEnclosingCircle(cnt)

        area_rect = rect[1][0] * rect[1][1]
        area_circ = np.pi * np.square(circ[1])

        if area_rect > area_circ:

            x = np.int0(circ[0][0])
            y = np.int0(circ[0][1])
            rad = np.int0(circ[1])

            dep_counter = 0
            # row = random.sample((x - rad, x + rad), sample_size * 3)
            # col = random.sample((y - rad, y + rad), sample_size *)
            row = range(x - rad, x + rad)
            col = range(y - rad, y + rad)

            for ii in row:
                for jj in col:
                    dx = ii - x
                    dy = jj - y
                    distsq = dx * dx + dy * dy
                    if (distsq <= rad * rad):
                        dep_itr[dep_counter] = [ii, jj]
                        dep_counter = dep_counter + 1
        return dep_itr

list_of_contours = Parallel(n_jobs=4, backend="threading")\
                    (delayed(unwrap_self)(cnt) for cnt in zip([self]*len(contours),contours))
def unwrap_self(arg, **kwargs):
    return Vision.check_depth_contours(*arg, **kwargs)