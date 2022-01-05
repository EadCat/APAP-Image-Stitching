from stitching.homography import Homography
import numpy as np
import cv2
from tqdm import tqdm


class Apap(Homography):
    def __init__(self, opt, final_size, offset):
        """
        :param opt: Engine running Options
        :param final_size: final result (Stitched) image size
        :param offset: The extent to which the image is stretched
        """
        super().__init__()
        self.gamma = opt.gamma
        self.sigma = opt.sigma
        self.final_width, self.final_height = final_size
        self.offset_x, self.offset_y = offset

    def global_homography(self, src_point, dst_point):
        raise NotImplementedError

    def local_homography(self, src_point, dst_point, vertices):
        """
        local homography estimation
        :param src_point: shape [sample_n, 2]
        :param dst_point:
        :param vertices: shape [mesh_size, mesh_size, 2]
        :return: np.ndarray [meshsize, meshsize, 3, 3]
        """
        sample_n, _ = src_point.shape
        mesh_n, pt_size, _ = vertices.shape

        N1, nf1 = self.getNormalize2DPts(src_point)
        N2, nf2 = self.getNormalize2DPts(dst_point)

        C1 = self.getConditionerFromPts(nf1)
        C2 = self.getConditionerFromPts(nf2)

        cf1 = self.point_normalize(nf1, C1)
        cf2 = self.point_normalize(nf2, C2)

        inverse_sigma = 1. / (self.sigma ** 2)
        local_homography_ = np.zeros([mesh_n, pt_size, 3, 3], dtype=np.float)
        local_weight = np.zeros([mesh_n, pt_size, sample_n])
        aa = self.matrix_generate(sample_n, cf1, cf2)  # initiate A

        for i in range(mesh_n):
            for j in range(pt_size):
                dist = np.tile(vertices[i, j], (sample_n, 1)) - src_point
                weight = np.exp(-(np.sqrt(dist[:, 0] ** 2 + dist[:, 1] ** 2) * inverse_sigma))
                weight[weight < self.gamma] = self.gamma
                local_weight[i, j, :] = weight
                A = np.expand_dims(np.repeat(weight, 2), -1) * aa
                W, U, V = cv2.SVDecomp(A)
                h = V[-1, :]
                h = h.reshape((3, 3))
                h = np.linalg.inv(C2).dot(h).dot(C1)
                h = np.linalg.inv(N2).dot(h).dot(N1)
                h = h / h[2, 2]
                local_homography_[i, j] = h
        return local_homography_, local_weight

    @staticmethod
    def warp_coordinate_estimate(pt, homography):
        """
        source points -> target points matrix multiplication with homography
        [ h11 h12 h13 ] [x]   [x']
        [ h21 h22 h23 ] [y] = [y']
        [ h31 h32 h33 ] [1]   [s']
        :param pt: source point
        :param homography: transfer relationship
        :return: target point
        """
        target = np.matmul(homography, pt)
        target /= target[2]
        return target

    def local_warp(self, ori_img: np.ndarray, local_homography: np.ndarray, mesh: np.ndarray,
                   progress=False) -> np.ndarray:
        """
        this method requires improvement with the numpy algorithm (because of speed)

        :param ori_img: original input image
        :param local_homography: [mesh_n, pt_size, 3, 3] local homographies np.ndarray
        :param mesh: [2, mesh_n+1]
        :param progress: print warping progress or not.
        :return: result(warped) image
        """
        mesh_w, mesh_h = mesh
        ori_h, ori_w, _ = ori_img.shape
        warped_img = np.zeros([self.final_height, self.final_width, 3], dtype=np.uint8)

        for i in tqdm(range(self.final_height)) if progress else range(self.final_height):
            m = np.where(i < mesh_h)[0][0]
            for j in range(self.final_width):
                n = np.where(j < mesh_w)[0][0]
                homography = np.linalg.inv(local_homography[m-1, n-1, :])
                x, y = j - self.offset_x, i - self.offset_y
                source_pts = np.array([x, y, 1])
                target_pts = self.warp_coordinate_estimate(source_pts, homography)
                if 0 < target_pts[0] < ori_w and 0 < target_pts[1] < ori_h:
                    warped_img[i, j, :] = ori_img[int(target_pts[1]), int(target_pts[0]), :]

        return warped_img





