# custom libraries
from feature.sift import SIFTMatcher
from feature.ransac import RANSAC
from stitching.homography import Homography, final_size
from stitching.apap import Apap
from stitching.blend import uniform_blend
from utils.mesh import get_mesh, get_vertice
from utils.draw import *
from utils.recursive import RecursiveDivider

# basic libraries
import time
import os
from PIL import Image


"""
***  Conventional Image-Stitching Pipeline ***
1. image load
2. grayscaling & SIFT
3. Brute-force MATCHING
4. RANSAC
5. Estimate Global-Homography & extract Final size
6. Estimate Local-Homograhpy
7. Warping
8. Blending
"""


class Thread:
    def __init__(self, opt):
        self.opt = opt
        self.n = 0  # line number of txt file
        self.unit_w, self.unit_h = self.opt.resize

    def recursive(self, imgdir):
        if isinstance(imgdir, list):
            if len(imgdir) == 2:
                return self.thread(self.recursive(imgdir[0]), self.recursive(imgdir[1]))
            else:
                return self.recursive(imgdir[0])
        else:
            src = cv2.imread(imgdir[0], cv2.IMREAD_COLOR)
            assert src is not None, print(f'No such directory exists:{imgdir[0]}')
            src = src[:, :, ::-1]
            src = cv2.resize(src, dsize=(self.unit_w, self.unit_h))
            try:
                # process stitching
                dst = cv2.imread(imgdir[1], cv2.IMREAD_COLOR)[:, :, ::-1]
                dst = cv2.resize(dst, dsize=(self.unit_w, self.unit_h))
                return self.thread(src, dst)
            except:
                # just return
                return src

    def process(self, imgdir, mask):
        unit_start = time.perf_counter()
        if self.opt.print_n: print(f'processing {self.n + 1} thread...')
        # ========================================== call image & stitch ==============================================
        result = self.recursive(imgdir)
        # ==============================================================================================================
        # ====================================== mask carving on result image ==========================================
        if isinstance(mask, str):
            mask_img = cv2.imread(mask, cv2.IMREAD_COLOR)[:, :, ::-1]
            mask_img = cv2.normalize(mask_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            h, w, _ = mask_img.shape
            result = cv2.resize(result, dsize=(w, h))
            result *= mask_img
        elif isinstance(mask, list):
            mask_img = cv2.imread(mask[self.n], cv2.IMREAD_COLOR)[:, :, ::-1]
            mask_img = cv2.normalize(mask_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            h, w, _ = mask_img.shape
            result = cv2.resize(result, dsize=(w, h))
            result *= mask_img
        else:
            pass
        # ==============================================================================================================
        # =========================================== save result image ================================================
        if self.opt.saveroot is not None:
            os.makedirs(self.opt.saveroot, exist_ok=True)
            save_dir = os.path.join(self.opt.saveroot, self.opt.savename +
                                    str(self.n).zfill(5) + '.' + self.opt.savefmt)
            cv2.imwrite(save_dir, result[:, :, ::-1])
            if self.opt.saveprint: print(f'{self.n+1} image saved -> {save_dir}')
        # ==============================================================================================================
        # =========================================== print result image ===============================================
        if self.opt.image_show > self.n or self.opt.image_show == -1:
            result = Image.fromarray(result)
            result.show()
        # ==============================================================================================================
        if self.opt.unit_time: print(f'{self.n + 1} image time spending: {time.perf_counter() - unit_start:4f}s.')
        self.n += 1

    @staticmethod
    def call_dataset(fname, root=None):
        file = open(fname, 'r')
        data = file.readlines()
        target_stack = []
        for d in data:
            imgname = d.strip().strip('\n').split(' ')
            if root is not None:
                # path merging
                target_stack.append([os.path.join(root, name) for name in imgname])
            else:
                # absolute path
                target_stack.append(imgname)
        return target_stack

    def call_mask(self):
        if self.opt.mask_dir is None:
            return None
        try:  # only one mask
            mask = self.opt.mask_dir
        except:  # mask text file
            mask = self.call_dataset(self.opt.mask_dir, root=self.opt.mask_root)
        return mask

    def thread(self, src, dst):
        mesh_size = self.opt.mesh_size

        img1 = src
        img2 = dst
        ori_h, ori_w, _ = img1.shape
        dst_h, dst_w, _ = img2.shape
        sift = SIFTMatcher()

        if self.opt.verbose: print(f'{self.n + 1} image SIFT...')
        # SIFT & KNN BFMatching
        src_match, dst_match = sift.thread(img1, img2)
        if self.opt.verbose: print(f"raw matching points: {len(src_match)}")

        # RANSAC
        if self.opt.verbose: print(f'{self.n + 1} image RANSAC...')
        ransac = RANSAC(self.opt)
        final_src, final_dst = ransac.thread(src_match, dst_match, self.opt.ransac_max)
        if self.opt.verbose: print(f'final matching points: {len(final_src)}')

        # Global Homography
        if self.opt.verbose: print(f'{self.n + 1} image Global Homography Estimation...')
        h_agent = Homography()
        gh = h_agent.global_homography(final_src, final_dst)
        final_w, final_h, offset_x, offset_y = final_size(img1, img2, gh)

        if final_h > ori_h * 4. or final_w > ori_w * 4.:
            print("Homography Estimation Failed.")
            final_h = max(ori_h, dst_h)
            final_w = max(ori_w, dst_w)
            result = np.zeros(shape=(final_w, final_h), dtype=np.uint8)
            img1 = cv2.resize(img1, dsize=(final_w, final_h))
            img2 = cv2.resize(img2, dsize=(final_w, final_h))
            result[:, :int(final_w/2), :] = img1[:, :int(final_w/2), :]
            result[:, int(final_w/2):, :] = img2[:, int(final_w/2):, :]
        else:
            # APAP
            # ready meshgrid
            mesh = get_mesh((final_w, final_h), mesh_size + 1)
            vertices = get_vertice((final_w, final_h), mesh_size, (offset_x, offset_y))

            # As-Projective-As-Possible Stitcher instance definition
            stitcher = Apap(self.opt, [final_w, final_h], [offset_x, offset_y])
            # local homography estimating
            if self.opt.verbose: print(f'{self.n+1} image local homography Estimation...')
            local_homography, local_weight = stitcher.local_homography(final_src, final_dst, vertices)
            # local warping
            if self.opt.verbose: print(f'{self.n+1} image local warping...')
            warped_img = stitcher.local_warp(img1, local_homography, mesh, self.opt.warping_progress)

            # another image pixel move
            dst_temp = np.zeros_like(warped_img)
            dst_temp[offset_y: dst_h + offset_y, offset_x: dst_w + offset_x, :] = img2

            # Uniform(50:50) blending
            if self.opt.verbose: print(f'{self.n+1} image blending...')
            result = uniform_blend(warped_img, dst_temp)

            # Draw
            if self.opt.match_print:
                match_fig = draw_match(img1, img2, final_src, final_dst, self.opt.matching_line)
                Image.fromarray(match_fig).show()

        # store
        return result

    def thread_choice(self):
        # mask setting
        mask = self.call_mask()
        # divider instance
        divider = RecursiveDivider()
        # two image stitching
        if None not in [self.opt.img1, self.opt.img2]:
            data = divider.list_divide([self.opt.img1, self.opt.img2])
            self.process(data, mask)
        # multi image stitching
        elif self.opt.imgs is not None:
            data = divider.list_divide(self.opt.imgs)
            self.process(data, mask)
        # image (root + txt list merging) or (absolute) path stitching
        elif None not in [self.opt.imgroot, self.opt.imglist]:
            datalist = self.call_dataset(self.opt.imglist, root=self.opt.imgroot)
            for data in datalist:
                data = divider.list_divide(data)
                self.process(data, mask)
                # self.process(data, mask)
        # error
        else:
            print('please enter input options.')
