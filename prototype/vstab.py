#!/usr/bin/env python3

import os
import argparse
import numpy as np
import cv2

DATA_FOLDER = os.path.join('..', 'data')
DEFAULT_VIDEO = 'pan.avi'
DETECTOR = cv2.xfeatures2d.SIFT_create()

def parse_args():
    parser = argparse.ArgumentParser(description='Stabilize a given video.')
    parser.add_argument('file', nargs='?', type=str, help='Video file inside data folder that shall be processed.')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    if args.file is None:
        args.file = DEFAULT_VIDEO
    args.file = os.path.join(DATA_FOLDER, args.file)
    return args

def read_video(file):
    video = cv2.VideoCapture(file)
    if not video.isOpened():
        print("Error: Could not open video '{}'.".format(file))
        return None
    frames = []
    while True:
        okay, frame = video.read()
        if not okay:
            break
        frames.append(frame)
    video.release()
    return frames

def to_int_tuple(pt):
    lst = np.int32(pt).tolist()
    return tuple(*lst)

def extract_cropped_rectangle(frame, tf):
    height, width, channels = frame.shape
    corners = np.array([[0, height], [width, height], [width, 0], [0, 0]], dtype=np.float32)
    corners = np.expand_dims(corners, 0)
    corners_tfed = cv2.perspectiveTransform(corners, tf)
    corners_tfed = np.squeeze(corners_tfed, axis=0)
    top = np.max(corners_tfed[[2, 3], 1], axis=0)
    bottom = np.min(corners_tfed[[0, 1], 1], axis=0)
    right = np.min(corners_tfed[[1, 2], 0], axis=0)
    left = np.max(corners_tfed[[3, 0], 0], axis=0)
    corners_cropped = np.array([[left, bottom], [right, bottom], [right, top], [left, top]])
    return corners_cropped

def main():
    args = parse_args()

    # Input
    print("Reading video...")
    frames = read_video(args.file)
    if frames is None:
        exit(1)

    # Process
    print("Detecting features...")
    transformations = [np.identity(3) for _ in range(len(frames))]
    for i in range(len(frames) - 1):
        frame_current = frames[i]
        frame_next = frames[i + 1]
        keypoints_current, descriptors_current = DETECTOR.detectAndCompute(frame_current, None)
        keypoints_next, descriptors_next = DETECTOR.detectAndCompute(frame_next, None)

        matcher = cv2.BFMatcher()
        all_matches = matcher.knnMatch(descriptors_current, descriptors_next, k = 2)

        good_matches = []
        for m, n in all_matches:
            # Only use match if it is significantly better than next best match.
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        pts_current = np.float32([ keypoints_current[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts_next = np.float32([ keypoints_next[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Debug viz
        if args.debug:
            for pt_current, pt_next in zip(pts_current, pts_next):
                pt_src, pt_dest = to_int_tuple(pt_current), to_int_tuple(pt_next)
                cv2.arrowedLine(frame_current, pt_src, pt_dest, (255, 120, 120))

        # Estimate homography
        homography, mask = cv2.findHomography(pts_next, pts_current, cv2.RANSAC)
        #homography[0, 2] = 0 # Stabilize only y axis.
        transformations[i + 1] = np.matmul(transformations[i], homography)

    centers = []
    for i, frame, tf in zip(range(len(frames)), frames, transformations):
        corners_cropped = extract_cropped_rectangle(frame, tf)
        min = np.min(corners_cropped, axis=0)
        max = np.max(corners_cropped, axis=0)
        center = (min + max) / 2
        centers.append(center)

    for i, frame, tf in zip(range(len(frames)), frames, transformations):
        height, width, channels = frame.shape
        frame = cv2.warpPerspective(frame, tf, (width, height))
        frames[i] = frame

    # Debug viz homography centers.
    if args.debug:
        for i, frame, center in zip(range(len(frames)), frames, centers):
            center_int = center.astype(np.int32)
            for j in range(i, len(frames)):
                cv2.circle(frames[j], (center_int[0], center_int[1]), 2, (120, 255, 120))

    # Display frame
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    i = 0
    while True:
        f = frames[i]
        cv2.imshow('frame', f)
        key = cv2.waitKey()
        if key == 27:
            break
        if key == 0x6a:
            # j
            i = (i + 1) % len(frames)
        if key == 0x6B:
            # k
            i = (i - 1) % len(frames)

    # Set down
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
