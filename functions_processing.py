import cv2
import numpy as np


def compute_distance(bboxes, H, F):
    """
    function: compute distance from person to camera
    :param bboxes: bounding box of person in image
    :return: x, y, z (cm)
    """
    x1, y1, x2, y2 = bboxes
    # center coordinate of bounding box
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    # height if person (pixel)
    height = y2 - y1
    # distance from camera base on triangle similarity technique (cm)
    distance = (H*F)/height

    # center coordinate of bounding boxes base on triangle similarity technique (cm)
    x_center_cm = (x_center * distance)/F
    y_center_cm = (y_center * distance)/F
    return x_center_cm, y_center_cm, distance


def get_transform(bbox, transform):
    """
    function: transform pixel
    bbox: bbox of object
    transform: matrix transform
    """
    bottem_point = ((bbox[0] + bbox[2])/2, bbox[3])
    pts = np.float32(np.array([[bottem_point]]))
    dst = cv2.perspectiveTransform(pts, transform)[0][0]
    return int(dst[0]), int(dst[1])


def point_distance(p1, p2, ratio_w, ratio_h, cm_w, cm_h):
    """
    Function: compute distance between two point unit cm
    p1: coordinate 1
    p2: coordinate 2
    ratio_w: number pixels on cm_w of width
    ratio_h: number pixels on cm_h of height
    cm_w: width (cm)
    cm_h: height (cm)
    """
    # compute w, h in pixel
    w = abs(p1[0] - p2[0])
    h = abs(p1[1] - p2[1])
    # transform unit pixel about unit cm
    w = w*(cm_w/ratio_w)
    h = h*(cm_h/ratio_h)
    return int(np.sqrt(w**2 + h**2))


def check_point_in_polygon(bbox, list_point):
    """
    function: check point in polygon
    bbox: bbox of person, ...
    list_point: list contain coordinate of polygon
    """
    bottem_point = ((bbox[0] + bbox[2])/2, bbox[3])
    sum_angle = 0
    for idx in range(len(list_point)):
        if idx < len(list_point) - 1:
            vector_a = np.array(bottem_point) - np.array(list_point[idx])
            vector_b = np.array(bottem_point) - np.array(list_point[idx + 1])
        else:
            vector_a = np.array(bottem_point) - np.array(list_point[idx])
            vector_b = np.array(bottem_point) - np.array(list_point[0])
        unit_vector_a = vector_a/np.linalg.norm(vector_a)
        unit_vector_b = vector_b/np.linalg.norm(vector_b)
        dot = np.dot(unit_vector_a, unit_vector_b)
        sum_angle += np.rad2deg(np.arccos(dot))
    return round(sum_angle)
