import cv2
import random
import numpy as np


def draw(src, src_point):
    """
    draw feature points on source image
    """
    dst = src.copy()
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)
    for i in range(src_point.shape[0]):
        cv2.circle(dst, (int(src_point[i, 0]), int(src_point[i, 1])), 5, color=(b, g, r), thickness=1)
    return dst


def draw_match(src, dst, src_point, dst_point, number=20):
    """
    draw matching lines between two images
    :param src: image 1
    :param dst: image 2
    :param src_point: matching point on image 1
    :param dst_point: matching point on image 2
    :param number: # of matching line
    :return: Drawn image
    """
    src_point = src_point.astype(np.int)
    dst_point = dst_point.astype(np.int)
    src_h, src_w, _ = src.shape
    dst_h, dst_w, _ = dst.shape
    final_height = max(src_h, dst_h)
    final_width = src_w + dst_w

    # copy
    pic = np.zeros([final_height, final_width, 3], dtype=np.uint8)
    pic[:src_h, :src_w, :] = src[:, :, :]
    pic[:dst_h, src_w:, :] = dst[:, :, :]

    # give destination offset
    dst_point[:, 0] = dst_point[:, 0] + src_w

    # matching #
    n = src_point.shape[0]
    if number > n:
        number = n

    # matching line draw
    for i in range(number):
        cv2.line(pic, (src_point[i, 0], src_point[i, 1]), (dst_point[i, 0], dst_point[i, 1]),
                 (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), thickness=1, lineType=cv2.LINE_AA)
    return pic
