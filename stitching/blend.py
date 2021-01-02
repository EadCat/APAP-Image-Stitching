import numpy as np


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
