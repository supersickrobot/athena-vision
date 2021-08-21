#             x0 = box[0][0]
#             y0 = box[0][1]
#             x1 = box[1][0]
#             y1 = box[1][1]
#             x2 = box[2][0]
#             y2 = box[2][1]
#             x3 = box[3][0]
#             y3 = box[3][1]
#
#             dep_itr = np.zeros(oi_config['area']['high']+1)
#             dep_counter = 0
#             # if x0 == x3 or y3 == y0:
#             #     row = range(x0, x2)
#             #     col = range(y1, y3)
#             #     for ii in row:
#             #         for jj in col:
#             #             dep_itr[dep_counter] = (depth_frame.as_depth_frame().get_distance(ii, jj))
#             # else:
#             #     slope1 = ((y3 - y0) / (x3 - x0))
#             #     slope2 = (x1 - x0)/ (y1 - y0)
#             #     intercept1 = y0 - x0 * slope1
#             #     intercept2 = y2 - x2 * slope1
#             #     intercept3 = x0 - y0 * slope2
#             #     intercept4 = x2 - y2 * slope2
#             #     if x0 < 0:
#             #         x0 = 0
#             #     if x2 > depth.shape[1]:
#             #         x2 = depth.shape[1]
#             #     if y1 < 0:
#             #         y1 = 0
#             #     if y3 > depth.shape[0]:
#             #         y3 = depth.shape[0]
#             #     row = range(x0, x2)
#             #     col = range(y1, y3)
#             #     for ii in row:
#             #         for jj in col:
#             #             check1 = slope1*ii+intercept1
#             #             check2 = slope1 * ii + intercept2
#             #             check3 = slope2*jj+intercept3
#             #             check4 = slope2*jj+intercept4
#             #             if check1 >= jj >= check2 and \
#             #                     check3 <= ii <= check4:
#             #                 dep_itr[dep_counter] = (depth_frame.as_depth_frame().get_distance(ii, jj))
#             #                 dep_counter = dep_counter + 1
#
#             dep_itr = dep_itr[dep_itr != 0]
#             dep_std = np.std(dep_itr)
#             dep_mean = np.mean(dep_itr)
#             depth_val = [a for a in dep_itr if (a > dep_mean - 2 * dep_std)]
#             depth_val = [a for a in depth_val if (a < dep_mean + 2 * dep_std)]
#             depth_val = np.mean(depth_val)