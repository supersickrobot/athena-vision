import pyrealsense2 as rs
import numpy as np
import cv2


def setup_pipe(config, device_id):
    pipeline = rs.pipeline()
    _config = rs.config()
    _config.enable_device(device_id)
    set_d = config['depth']
    set_c = config['color']
    _config.enable_stream(rs.stream.depth, set_d["resolution"][0], set_d["resolution"][1], rs.format.z16, set_d['fps'])
    _config.enable_stream(rs.stream.color, set_c["resolution"][0], set_c["resolution"][1], rs.format.bgr8, set_c['fps'])

    profile = pipeline.start(_config)

    return pipeline, profile


def aligned(config, frames):
    color_frame = frames.get_color_frame()
    align = rs.align(rs.stream.color)
    frameset = align.process(frames)
    aligned_depth_frame = frameset.get_depth_frame()

    colorizer = rs.colorizer()
    push_depth = np.asanyarray(aligned_depth_frame.get_data())

    colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
    color = np.asanyarray(color_frame.get_data())
    alpha = config['alpha']
    beta = 1 - alpha
    images = cv2.addWeighted(color, alpha, colorized_depth, beta, 0.0)

    set_e = config['edge']
    low_threshold = set_e['low_threshold']
    ratio = set_e['ratio']  # recommend ratio 2:1 - 3:1
    kernal = set_e['kernal']
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    depth_to_jet = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    edges = cv2.Canny(depth_to_jet, low_threshold, low_threshold * ratio, kernal)
    # images = depth_image
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    images = cv2.addWeighted(images, alpha, edges, beta, 0.0)

    return images, push_depth, color, edges


def find_background(depth, num_bins, mult):
    depth_data = depth.copy()
    # depth_data = np.asanyarray(depth.get_data())
    hist, bins = np.histogram(depth_data[depth_data != 0], bins=num_bins)
    high_bin = bins[np.argmax(hist)]
    bin_width = (np.amax(bins) - np.amin(bins)) / num_bins
    dd = depth_data

    low = high_bin-bin_width*(mult-1)
    high = high_bin+bin_width*mult
    dd[dd > high] = 0
    dd[dd < low] = 0

    tbl_height = high_bin

    binary = dd.copy()
    binary[binary != 0] = 1
    # binary = cv2.applyColorMap(cv2.convertScaleAbs(binary, alpha=0.03), cv2.COLORMAP_JET)
    return dd, binary, tbl_height


def isolate_background(depth, height, tolerance_near, tolerance_far):
    depth_data = depth.copy()
    far = height+tolerance_far
    near = height-tolerance_near
    depth_data[depth_data < near] = 0
    depth_data[depth_data > far] = 0
    return depth_data


def xyz(depth):
    pc = rs.pointcloud()
    points = pc.calculate(depth)
    pts = np.asanyarray(points.get_vertices())
    ap = [*zip(*pts)]
    x = ap[0]
    y = ap[1]
    z = ap[2]

    return x, y, z


def centroid(binary):
    flat = binary.sum(axis=1)
    peak_thresh=100
    min_width = 50
    counter = 0
    peaks =[]
    depths=[]
    for i, value in enumerate(flat):
        if value < peak_thresh:
            if counter >min_width:
                peaks.append(i-counter/2)
                depths.append(counter)
            counter = 0
        else:
            counter = counter+1

    ind = depths.index(max(depths))
    return peaks[ind], depths[ind]


def crop_horiz(image, center, width):
    l_bound = int(center-width/2)
    u_bound = int(center+width/2)
    out = image[:, l_bound:u_bound]
    return out, l_bound, u_bound


def crop_vert(image, center, depth):
    l_bound = int(center-depth/2)
    u_bound = int(center+depth/2)
    out = image[l_bound:u_bound, :]
    return out, l_bound, u_bound

def crop_below_workspace(image, table_edge):
    u_bound = image.shape[0]
    l_bound = int(table_edge)
    out = image[l_bound: u_bound, :]
    return out

def std_filter(frame):
    dec_filter = rs.decimation_filter()
    spat_filter = rs.spatial_filter()
    temp_filter = rs.temporal_filter()

    filtered = dec_filter.process(frame)
    filtered = spat_filter.process(filtered)
    filtered = temp_filter.process(filtered)
    return filtered
# cv2.polylines