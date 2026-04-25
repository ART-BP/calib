#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import cv2
import numpy as np


def get_go2_front_fisheye_params():
    # camera_name: go2_front_fisheye
    k = np.array(
        [
            [1203.762044004368, 0.0, 981.7904792654031],
            [0.0, 1203.7009720218682, 525.2625697472332],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    d = np.array(
        [
            -0.06940178268945467,
            -0.05259276838826166,
            0.060392401913685174,
            -0.03652503468416535,
        ],
        dtype=np.float64,
    ).reshape(4, 1)
    return k, d


def undistort_fisheye(image: np.ndarray, k: np.ndarray, d: np.ndarray, balance: float):
    h, w = image.shape[:2]
    r = np.eye(3, dtype=np.float64)

    # Compute a valid new intrinsics matrix for rectified output.
    new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K=k,
        D=d,
        image_size=(w, h),
        R=r,
        balance=balance,
        new_size=(w, h),
        fov_scale=1.0,
    )

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K=k,
        D=d,
        R=r,
        P=new_k,
        size=(w, h),
        m1type=cv2.CV_16SC2,
    )

    undistorted = cv2.remap(
        image,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return undistorted, new_k


def main():
    parser = argparse.ArgumentParser(
        description="Read an image, undistort with go2_front_fisheye intrinsics, and save it."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="mid.png",
        help="Input image path",
    )
    parser.add_argument(
        "--output",
        "-o",
         default="mid_undis0.png",
        help="Output image path",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=0.0,
        help="0.0~1.0. Smaller means less black border and narrower FOV; larger keeps more FOV.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {input_path}")

    k, d = get_go2_front_fisheye_params()
    undistorted, new_k = undistort_fisheye(image, k, d, balance=args.balance)

    ok = cv2.imwrite(str(output_path), undistorted)
    if not ok:
        raise RuntimeError(f"Failed to save image: {output_path}")

    np.set_printoptions(precision=10, suppress=True)
    print(f"[Done] Saved undistorted image to: {output_path}")
    print("[Info] New camera matrix (for undistorted image):")
    print(new_k)


if __name__ == "__main__":
    main()
