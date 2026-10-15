import numpy as np
import cv2


def uniform_blend(img1, img2):
    # grayscale
    gray1 = np.mean(img1, axis=-1)
    gray2 = np.mean(img2, axis=-1)
    result = (img1.astype(np.float64) + img2.astype(np.float64))

    g1, g2 = gray1 > 0, gray2 > 0
    g = g1 & g2
    mask = np.expand_dims(g * 0.5, axis=-1)
    mask = np.tile(mask, [1, 1, 3])
    mask[mask == 0] = 1
    result *= mask
    result = result.astype(np.uint8)

    return result


def multiband_blend(img1, img2, levels=5):
    """
    Burt & Adelson multi-band blending with Laplacian pyramids
    :param img1: warped image [h, w, 3]
    :param img2: another image on the same canvas [h, w, 3]
    :param levels: # of pyramid levels
    :return: blended image
    """
    g1, g2 = np.mean(img1, axis=-1) > 0, np.mean(img2, axis=-1) > 0

    # seam: each overlapped pixel belongs to the image whose border is farther
    d1 = cv2.distanceTransform(g1.astype(np.uint8), cv2.DIST_L2, 5)
    d2 = cv2.distanceTransform(g2.astype(np.uint8), cv2.DIST_L2, 5)
    weight = (d1 > d2).astype(np.float32)

    # fill the empty area with the other image so that only the seam is blended, not the black background
    img1 = np.where(np.expand_dims(g1, -1), img1, img2).astype(np.float32)
    img2 = np.where(np.expand_dims(g2, -1), img2, img1).astype(np.float32)

    gp1, gp2, gpw = [img1], [img2], [weight]
    for _ in range(levels):
        gp1.append(cv2.pyrDown(gp1[-1]))
        gp2.append(cv2.pyrDown(gp2[-1]))
        gpw.append(cv2.pyrDown(gpw[-1]))

    w = np.expand_dims(gpw[-1], -1)
    result = w * gp1[-1] + (1 - w) * gp2[-1]
    for k in range(levels - 1, -1, -1):
        size = (gp1[k].shape[1], gp1[k].shape[0])
        lap1 = gp1[k] - cv2.pyrUp(gp1[k + 1], dstsize=size)
        lap2 = gp2[k] - cv2.pyrUp(gp2[k + 1], dstsize=size)
        w = np.expand_dims(gpw[k], -1)
        result = cv2.pyrUp(result, dstsize=size) + w * lap1 + (1 - w) * lap2

    result = np.clip(result, 0, 255).astype(np.uint8)
    result[~(g1 | g2)] = 0

    return result
