# Multiple view geometry code
#
import numpy as np
import cv2


def order_points(pts):
    # Order points top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # Top left point: smallest sum, bottom right, largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # top-right: smallest difference, bottom-right, largest distance
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(img, pts):

    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Max width of new image is maximum of bottom-left-right and
    # top left-right via L2 distance

    width_t = np.linalg.norm(tr - tr)
    width_b = np.linalg.norm(br - bl)
    width = max(int(width_t), int(width_b))

    # Height of new image, maximum of left-top-bottom and right-top-bottom
    height_l = np.linalg.norm(tl - bl)
    height_r = np.linalg.norm(tr - br)
    height = max(int(height_l), int(height_r))

    # Destination square
    dst = np.array([
        [0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (width, height))

    # return the warped image
    return warped
