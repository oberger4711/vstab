# vstab
Video stabilizer that smoothes camera motion.

## Setup
Requirements:
* OpenCV built with contrib modules (this requires building it from source)
* Ceres
* Boost

Build it with cmake:
```
cd src
mkdir build
cd build
cmake ..
make
```

## Algorithm
This is the pipeline:
1. Detect keypoints and descriptors with SIFT in each frame.
2. Estimate homography transformation between two consecutive frames using RANSAC to find keypoint correspondencies.
The transformation can be undone which ideally results in video of no camera motion
3. Smoothen camera motion by regressing a translation for each frame using non-linear Least Squares.
The following two costs are minimized in the process:
    1. The difference of the translation to the center of the frame after eliminating the camera motion.
    2. The difference in the steps from the translation of the previous frame and to the translation of the next frame.
4. Apply the transformation from 2. and 3.
5. Crop the frames to the largest rectangle with the original aspect ratio that always contains content.

A smoothing factor amplifies the costs of 3ii.
This is effectively a trade-of between a smooth camera motion and a low loss of pixels in the following cropping step.