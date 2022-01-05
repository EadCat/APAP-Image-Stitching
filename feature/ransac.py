import math
import numpy as np
import random
from stitching.homography import Homography


class RANSAC:
    def __init__(self, opt):
        self.opt = opt
        self.prob = opt.sample_inlier
        self.inlier_p = opt.ransac_inlier_prob
        self.threshold = opt.ransac_thres
        self.m = opt.ransac_sample  # # of sample for RANSAC algorithm
        if opt.optimal_ransac:
            self.N = min(self.get_N(), opt.ransac_max)  # optimal try number for RANSAC
        else:
            self.N = opt.ransac_max
        if opt.verbose:
            print(f"RANSAC trial optimization: {opt.optimal_ransac}")
            print(f"The number of RANSAC trials: {self.N}")

    def get_N(self):
        """
        calculate how much RANSAC to try.
        https://darkpgmr.tistory.com/61 -- (2)
        :return: # of Iteration
        """
        return int(round(math.log(1 - self.prob) / math.log(1 - math.pow((1 - self.inlier_p), self.m) + 1e-8)))

    @staticmethod
    def get_distance(src_point, dst_point, h):
        """
        The distance is calculated to estimate the optimal model.
        :param src_point:
        :param dst_point:
        :param h:
        :return:
        """
        x1, y1 = src_point
        x2, y2 = dst_point
        pt_query = np.transpose(np.array([x1, y1, 1.]))
        projection = np.dot(h, pt_query)
        projection /= (projection[2] + 1e-8)

        pt_train = np.transpose(np.array([x2, y2, 1.]))
        error = pt_train - projection

        return np.linalg.norm(error)

    def thread(self, src, dst, max_try=1000):
        """
        filter out bad matching feature points.
        :param src: matching feature points in source image
        :param dst: matching feature points in destination image
        :param max_try: user setting
        :return: good matching points
        """
        smax_inliers = []
        dmax_inliers = []
        arr_range = list(np.arange(len(src)))
        h_agent = Homography()

        for i in range(self.N):
            indices = random.sample(arr_range, self.m)

            # shape: (self.m, 2) [x, y]
            chosen_src = src[indices, :]
            chosen_dst = dst[indices, :]

            H = h_agent.global_homography(chosen_src, chosen_dst)

            src_inliers = []
            dst_inliers = []

            for s, d in zip(src, dst):
                dist = self.get_distance(s, d, H)
                if dist < self.threshold:
                    src_inliers.append(s)
                    dst_inliers.append(d)
            if len(src_inliers) > len(smax_inliers):
                smax_inliers = src_inliers
                dmax_inliers = dst_inliers

        return np.array(smax_inliers), np.array(dmax_inliers)



















