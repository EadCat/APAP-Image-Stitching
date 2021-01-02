import cv2
import numpy as np
"""
=================================================================================================
cv::KeyPoint Class Reference
 * Public Attributes
  - float       angle
    : computed orientation of the keypoint (-1 if not applicable); 
      it's in [0,360) degrees and measured relative to image coordinate system, ie in clockwise.
  - int         class_id
    : object class (if the keypoints need to be clustered by an object they belong to)
  - int         octave
    : octave (pyramid layer) from which the keypoint has been extracted
  - Point2f     pt
    : coordinates of the keypoints
  - float       response
    : the response by which the most strong keypoints have been selected. 
      Can be used for the further sorting or subsampling
  - float       size
    : diameter of the meaningful keypoint neighborhood
=================================================================================================
    
=================================================================================================
cv::DMatch Class Reference
 * Public Attributes
  - float       distance
  - int         imgIdx
    : train image index
  - int         queryIdx
    : query descriptor index
  - int         trainIdx
    : train descriptor index
=================================================================================================
"""


class Matcher:
    """
    https://stackoverflow.com/questions/10765066/what-is-query-and-train-in-opencv-features2d
    query: left
    train: right
    """
    def __init__(self):
        self.matcher = cv2.BFMatcher()
        self.raw_match = None
        self.goods = None

    def knn_match(self, desc1, desc2, k=2):
        # KNN Matching
        self.raw_match = self.matcher.knnMatch(desc1, desc2, k)
        return self.raw_match

    def good_matching(self, ratio=0.7):
        # filter good matching points
        self.goods = []
        for first, second in self.raw_match:
            if first.distance < second.distance * ratio:
                self.goods.append((first.trainIdx, first.queryIdx))
        return self.goods

    def form(self, kp1, kp2):
        if len(self.goods) > 4:
            psta = np.array([kp1[i].pt for _, i in self.goods])
            src_match = psta[:, :2]
            pstb = np.array([kp2[i].pt for i, _ in self.goods])
            dst_match = pstb[:, :2]
        else:
            src_match = np.zeros([1, 3])
            dst_match = np.zeros([])
        return src_match, dst_match # [len, 2], [len, 2]

    def detect(self, *args, **kwargs):
        raise NotImplementedError

    def raw_match(self):
        return self.raw_match

    def good_match(self):
        return self.goods

    def thread(self, img1, img2, ratio=0.7):
        kp1, desc1 = self.detect(img1)
        kp2, desc2 = self.detect(img2)
        self.knn_match(desc1, desc2)
        self.good_matching(ratio)
        src_match, dst_match = self.form(kp1, kp2)
        return src_match, dst_match


class SIFTMatcher(Matcher):
    def __init__(self):
        super().__init__()
        self.agent = cv2.xfeatures2d.SIFT_create()

    def detect(self, img):
        # return (kp, desc)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return self.agent.detectAndCompute(gray, None)
