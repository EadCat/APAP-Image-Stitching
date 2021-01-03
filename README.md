# As-Projective-As-Possible Image Stitching with Moving DLT

---

2020.12.28. ~ 2021.01.03.

Local-Homography warping

This program takes a list of images and performs stitching recursively.



## Target Research Paper

The research paper: https://cs.adelaide.edu.au/~tjchin/apap/

**Zaragoza, Julio, et al. "As-projective-as-possible image stitching with moving DLT." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2013.**

![Figure1](./assets/figure1.jpg)



## Dependencies

```
python == 3.8.5
numpy == 1.19.2
opencv-python == 4.4.0.46
opencv-contrib-python == 4.4.0.46
pillow == 8.0.1
argparse
```

Editor: PyCharm



## Quick Start

### Windows User

run demo.bat files on terminal.



### Linux User

unzip demo_sh.zip and run demo.sh files on terminal.



## Pipeline

1. Image Loading
2. Grayscaling & SIFT(OpenCV) 
3. Brute-Force Matching
4. RANSAC
5. Estimating Global-Homography & Extracting Final size
6. Estimating Local-Homography
7. Local Image Warping
8. Uniform Blending (50:50)



## Reference

### code

1. https://github.com/lxlscut/APAP_S
2. https://github.com/fredzzhang/Normalized-Eight-Point-Algorithm
3. https://cs.adelaide.edu.au/~tjchin/apap/#Source

### demo images

1. https://github.com/daeyun/Image-Stitching
2. https://github.com/opencv/opencv_extra
3. https://www.pyimagesearch.com/2018/12/17/image-stitching-with-opencv-and-python/



thanks.



## Author

Dae-Young Song

Undergraduate student, Department of Electronic Engineering, Chungnam National University

[Github][EadCat (Dae-Young Song) (github.com)](https://github.com/EadCat)

