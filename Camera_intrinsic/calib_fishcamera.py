#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import cv2
import yaml
import numpy as np

# =========================
# 1. 标定板参数
# =========================
pattern_size = (11, 8)   # 内角点数 (cols, rows)
square_size = 0.04       # 方格边长，单位: 米

image_dir = "./images"
image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) +
                     glob.glob(os.path.join(image_dir, "*.png")))

if len(image_paths) == 0:
    raise RuntimeError("未找到标定图片，请检查 ./images 目录")

# =========================
# 2. 构造棋盘格世界坐标
# fisheye 要求 shape 更严格
# objp: (1, N, 3)
# =========================
objp = np.zeros((1, pattern_size[0] * pattern_size[1], 3), np.float64)
objp[0, :, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints = []

criteria_subpix = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    1e-6
)

img_size = None
valid_count = 0

vis_dir = "./corner_vis_fisheye"
if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)

# =========================
# 3. 提取角点
# imgpoints for fisheye: (1, N, 2)
# =========================
for img_path in image_paths:
    img = cv2.imread(img_path)
    if img is None:
        print("[跳过] 无法读取: {}".format(img_path))
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img_size is None:
        img_size = gray.shape[::-1]   # (w, h)
    else:
        if gray.shape[::-1] != img_size:
            print("[跳过] 分辨率不一致: {}".format(img_path))
            continue

    ret, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
              cv2.CALIB_CB_NORMALIZE_IMAGE +
              cv2.CALIB_CB_FAST_CHECK
    )

    if not ret:
        print("[失败] 未检测到角点: {}".format(img_path))
        continue

    corners = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        criteria_subpix
    )

    objpoints.append(objp.copy())
    imgpoints.append(corners.reshape(1, -1, 2).astype(np.float64))
    valid_count += 1

    vis = img.copy()
    cv2.drawChessboardCorners(vis, pattern_size, corners, ret)
    cv2.imwrite(os.path.join(vis_dir, os.path.basename(img_path)), vis)

    print("[成功] {}".format(img_path))

if valid_count < 10:
    raise RuntimeError("有效图片太少，仅 {} 张，建议至少 15~20 张".format(valid_count))

print("\n有效标定图片数量: {}".format(valid_count))
print("图像分辨率: {} x {}".format(img_size[0], img_size[1]))

# =========================
# 4. fisheye 标定
# =========================
K = np.zeros((3, 3), dtype=np.float64)
D = np.zeros((4, 1), dtype=np.float64)

rvecs = []
tvecs = []

flags = (
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
    cv2.fisheye.CALIB_CHECK_COND +
    cv2.fisheye.CALIB_FIX_SKEW
)

criteria_calib = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    500,
    1e-8
)

rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
    objpoints,
    imgpoints,
    img_size,
    K,
    D,
    rvecs,
    tvecs,
    flags,
    criteria_calib
)

print("\n===== fisheye 标定结果 =====")
print("RMS reprojection error = {}".format(rms))
print("K =\n{}".format(K))
print("D =\n{}".format(D.ravel()))

# =========================
# 5. 手动计算平均重投影误差
# =========================
total_error = 0.0

for i in range(len(objpoints)):
    projected, _ = cv2.fisheye.projectPoints(
        objpoints[i],
        rvecs[i],
        tvecs[i],
        K,
        D
    )
    error = cv2.norm(imgpoints[i], projected, cv2.NORM_L2) / projected.shape[1]
    total_error += error

mean_error = total_error / len(objpoints)
print("Mean reprojection error = {}".format(mean_error))

# =========================
# 6. 去畸变测试
# balance=0 更裁边、直线更直
# balance=1 视野更大
# 先取一个中间值
# =========================
for balance in list([0, 1]):
    test_img = cv2.imread(image_paths[0])
    h, w = test_img.shape[:2]
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=balance
    )

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
    )

    undistorted = cv2.remap(
        test_img,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    cv2.imwrite(f"undistorted_fisheye_{str(balance)}.png", undistorted)

# =========================
# 7. 保存 OpenCV 参数
# =========================
np.savez(
    "calib_result_fisheye.npz",
    K=K,
    D=D,
    image_width=w,
    image_height=h,
    new_K=new_K
)
print("已保存: calib_result_fisheye.npz")

# =========================
# 8. 保存 YAML
# 注意：这是 fisheye 参数，不建议伪装成 plumb_bob
# =========================
yaml_data = {
    "image_width": int(w),
    "image_height": int(h),
    "camera_name": "go2_front_fisheye",
    "camera_matrix": {
        "rows": 3,
        "cols": 3,
        "data": K.reshape(-1).tolist()
    },
    "distortion_model": "fisheye",
    "distortion_coefficients": {
        "rows": 1,
        "cols": 4,
        "data": D.reshape(-1).tolist()
    },
    "rectification_matrix": {
        "rows": 3,
        "cols": 3,
        "data": np.eye(3).reshape(-1).tolist()
    },
    "projection_matrix": {
        "rows": 3,
        "cols": 4,
        "data": [
            float(new_K[0, 0]), 0.0, float(new_K[0, 2]), 0.0,
            0.0, float(new_K[1, 1]), float(new_K[1, 2]), 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
    }
}

with open("camera_intrinsics_fisheye.yaml", "w") as f:
    yaml.dump(yaml_data, f, sort_keys=False)

print("已保存: camera_intrinsics_fisheye.yaml")
print("已保存可视化角点目录: {}".format(vis_dir))