import numpy as np
import math
import copy, cv2


class Homography:
    def __init__(self):
        pass

    @staticmethod
    def getNormalize2DPts(point):
        """
        :param point: [sample_num, 2]
        :return:
        """
        sample_n, _ = point.shape
        origin_point = copy.deepcopy(point)
        # np.ones(6) [1, 1, 1, 1, 1, 1]
        padding = np.ones(sample_n, dtype=np.float)
        c = np.mean(point, axis=0)
        pt = point - c
        pt_square = np.square(pt)
        pt_sum = np.sum(pt_square, axis=1)
        pt_mean = np.mean(np.sqrt(pt_sum))
        scale = math.sqrt(2) / (pt_mean + 1e-8)
        # https://www.programmersought.com/article/9488862074/
        t = np.array([[scale, 0, -scale * c[0]],
                      [0, scale, -scale * c[1]],
                      [0, 0, 1]], dtype=np.float)
        origin_point = np.column_stack((origin_point, padding))
        new_point = t.dot(origin_point.T)
        new_point = new_point.T[:, :2]
        return t, new_point

    @staticmethod
    def getConditionerFromPts(point):
        sample_n, _ = point.shape
        calculate = np.expand_dims(point, 0)
        mean_pts, std_pts = cv2.meanStdDev(calculate)
        mean_x, mean_y = np.squeeze(mean_pts)
        std_pts = np.squeeze(std_pts)
        std_pts = std_pts * std_pts * sample_n / (sample_n - 1)
        std_pts = np.sqrt(std_pts)
        std_x, std_y = std_pts
        std_x = std_x + (std_x == 0)
        std_y = std_y + (std_y == 0)
        norm_x = math.sqrt(2) / std_x
        norm_y = math.sqrt(2) / std_y
        T = np.array([[norm_x, 0, (-norm_x * mean_x)],
                      [0, norm_y, (-norm_y * mean_y)],
                      [0, 0, 1]], dtype=np.float)

        return T

    @staticmethod
    def point_normalize(nf, c):
        sample_n, _ = nf.shape
        cf = np.zeros_like(nf)

        for i in range(sample_n):
            cf[i, 0] = nf[i, 0] * c[0, 0] + c[0, 2]
            cf[i, 1] = nf[i, 1] * c[1, 1] + c[1, 2]

        return cf

    @staticmethod
    def matrix_generate(sample_n, cf1, cf2):
        A = np.zeros([sample_n * 2, 9], dtype=np.float)
        for k in range(sample_n):
            A[2 * k, 0] = cf1[k, 0]
            A[2 * k, 1] = cf1[k, 1]
            A[2 * k, 2] = 1
            A[2 * k, 6] = (-cf2[k, 0]) * cf1[k, 0]
            A[2 * k, 7] = (-cf2[k, 0]) * cf1[k, 1]
            A[2 * k, 8] = (-cf2[k, 0])

            A[2 * k + 1, 3] = cf1[k, 0]
            A[2 * k + 1, 4] = cf1[k, 1]
            A[2 * k + 1, 5] = 1
            A[2 * k + 1, 6] = (-cf2[k, 1]) * cf1[k, 0]
            A[2 * k + 1, 7] = (-cf2[k, 1]) * cf1[k, 1]
            A[2 * k + 1, 8] = (-cf2[k, 1])
        return A

    def global_homography(self, src_point, dst_point):
        """
        get global homography
        This engine used this method to get the final size.
        :param src_point: source image
        :param dst_point: destination image
        :return: global homography
        """
        sample_n, _ = src_point.shape
        # point normalization
        N1, nf1 = self.getNormalize2DPts(src_point)
        N2, nf2 = self.getNormalize2DPts(dst_point)

        C1 = self.getConditionerFromPts(nf1)
        C2 = self.getConditionerFromPts(nf2)

        cf1 = self.point_normalize(nf1, C1)
        cf2 = self.point_normalize(nf2, C2)

        # x' = Ax, make transform matrix
        A = self.matrix_generate(sample_n, cf1, cf2)

        # Singular Value Decomposition
        W, U, V = cv2.SVDecomp(A)

        # get global-homography
        h = V[-1, :]
        h = h.reshape((3, 3))
        h = np.linalg.inv(C2).dot(h).dot(C1)
        h = np.linalg.inv(N2).dot(h).dot(N1)
        h = h / h[2, 2]
        return h

    # @staticmethod
    # def global_homography(src_point, dst_point):
    #     """
    #     :param src_point: np.ndarray [[x1, y1], [x1, y1], [x1, y1], ...]
    #     :param dst_point: np.ndarray [[x2, y2], [x2, y2], [x2, y2], ...]
    #     :return: np.ndarray Homography
    #     """
    #     AList = []
    #     for src, dst in zip(src_point, dst_point):
    #         x1 = src[0]; y1 = src[1]
    #         x2 = dst[0]; y2 = dst[1]
    #         """
    #         Theory
    #         [ -x, -y, -1, 0, 0, 0, xx', yx', x'
    #          0, 0, 0, -x, -y, -1, xy', yy', y' ]
    #         """
    #         A = np.array([[-x1, -y1, -1, 0, 0, 0, x1*x2, y1*x2, x2],
    #                      [0, 0, 0, -x1, -y1, -1, x1*y2, y1*y2, y2]])
    #         AList.append(A)
    #
    #     MatA = np.reshape(np.array(AList), (-1, 9))
    #     # SVD Composition
    #     U, S, V = np.linalg.svd(MatA)
    #
    #     H = np.reshape(V[-1, :], (3, 3))
    #
    #     # normalize
    #     H = (1 / H[-1]) * H
    #
    #     return H


def final_size(src_img, dst_img, project_H):
    """
    get the size of stretched (stitched) image
    :param src_img: source image
    :param dst_img: destination image
    :param project_H: global homography
    :return:
    """
    h, w, c = src_img.shape

    corners = []
    pt_list = [np.array([0, 0, 1], dtype=np.float64), np.array([0, h, 1], dtype=np.float64),
               np.array([w, 0, 1], dtype=np.float64), np.array([w, h, 1], dtype=np.float64)]

    for pt in pt_list:
        vec = np.matmul(project_H, pt)
        x, y = vec[0] / vec[2], vec[1] / vec[2]
        corners.append([x, y])

    corners = np.array(corners).astype(np.int)

    h, w, c = dst_img.shape

    max_x = max(np.max(corners[:, 0]), w)
    max_y = max(np.max(corners[:, 1]), h)
    min_x = min(np.min(corners[:, 0]), 0)
    min_y = min(np.min(corners[:, 1]), 0)

    width = max_x - min_x
    height = max_y - min_y
    offset_x = -min_x if min_x < 0 else 0
    offset_y = -min_y if min_y < 0 else 0

    return width, height, offset_x, offset_y