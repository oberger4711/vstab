# vstab
Video stabilizer that smoothes camera motion.

Raw:

![Raw](https://github.com/oberger4711/vstab/blob/master/images/raw.gif)

Smoothed:

![Smoothed](https://github.com/oberger4711/vstab/blob/master/images/smoothed.gif)

## Setup
Built on Linux but may work on windows (somehow).

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
Run to following to find out how to use it:
```
./vstab --help
```

## Algorithm
This is the pipeline:
1. Detect keypoints and descriptors with SIFT in each frame.
2. Estimate homography transformation between two consecutive frames using RANSAC to find keypoint correspondencies.
The transformation can be undone which ideally results in video of no camera motion.
3. Smoothen camera motion by regressing a translation for each frame using non-linear Least Squares.
The following two costs are minimized in the process:
    1. Centered: The difference of the translation to the center of the frame estimated using the keypoint correspondencies.
    2. Smoothed: The difference in the steps from the translation of the previous frame and to the translation of the next frame.
4. Apply the transformation from 2. and 3.
5. Crop the frames to the largest rectangle with the original aspect ratio that always contains content.

## Smoothing Parameter
The only parameter of the algorithm is a smoothing factor that amplifies the costs of 3ii.
This is effectively a trade-of between a smooth camera motion and a low loss of pixels in the following cropping step.
The user can tune it to achieve the desired strength of smoothing for his application.

## Example
This is shown in the following screenshot:
* Blue: Correspondencies between keypoints of the current and the following frame (RANSAC is robust against outliers)
* Green: Original camera motion trajectory estimated from keypoint correspondences
* Red: Smoothed camera motion trajectory minimizing the cost functions (smoothness controlled by the smoothing factor)

![Screenshot of smoothing](https://github.com/oberger4711/vstab/blob/master/images/smoothing.png)
