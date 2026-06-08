#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import os
import sys

import numpy as np
import rosbag
from sensor_msgs.msg import PointField


EXTRINSIC_T = np.array([0.171, 0.0, 0.0908], dtype=np.float64)
EXTRINSIC_R = np.array(
    [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "calib_data", "scene"))
DEFAULT_BAGS = ["mid.bag", "left.bag", "right.bag"]
POINTCLOUD2_TYPE = "sensor_msgs/PointCloud2"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Read mid/left/right PointCloud2 rosbag files, transform lidar points "
            "to base_link with the given extrinsic, and write transformed bags."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Directory containing input bags. Defaults to FAST-Calib/calib_data/scene.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output bags. Defaults to the input bag directory.",
    )
    parser.add_argument(
        "--bags",
        nargs="+",
        default=DEFAULT_BAGS,
        help="Input bag names or absolute paths. Defaults to mid.bag left.bag right.bag.",
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        default=None,
        help="PointCloud2 topics to transform. Defaults to all PointCloud2 topics in each bag.",
    )
    parser.add_argument(
        "--output-topic",
        default=None,
        help="Optional output topic name for transformed clouds. Defaults to keeping input topic names.",
    )
    parser.add_argument(
        "--frame-id",
        default="base_link",
        help="Frame id written to transformed PointCloud2 headers.",
    )
    parser.add_argument(
        "--copy-other-topics",
        action="store_true",
        help="Copy non-selected topics into the output bag unchanged.",
    )
    parser.add_argument(
        "--translation",
        nargs=3,
        type=float,
        default=EXTRINSIC_T.tolist(),
        metavar=("TX", "TY", "TZ"),
        help="Translation from lidar frame to base_link.",
    )
    parser.add_argument(
        "--rotation",
        nargs=9,
        type=float,
        default=EXTRINSIC_R.reshape(-1).tolist(),
        metavar=("R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22"),
        help="Row-major 3x3 rotation from lidar frame to base_link.",
    )
    return parser.parse_args()


def resolve_bag_path(input_dir, bag_name):
    if os.path.isabs(bag_name):
        candidates = [bag_name]
    else:
        candidates = [os.path.join(input_dir, bag_name)]

    root, ext = os.path.splitext(bag_name)
    if ext == ".bag" and not root.endswith("_short"):
        fallback_name = root + "_short.bag"
        fallback = fallback_name if os.path.isabs(fallback_name) else os.path.join(input_dir, fallback_name)
        candidates.append(fallback)

    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

    tried = ", ".join(os.path.abspath(candidate) for candidate in candidates)
    raise FileNotFoundError("input bag not found: {} (tried: {})".format(bag_name, tried))


def get_pointcloud2_topics(bag):
    topic_info = bag.get_type_and_topic_info().topics
    return sorted(
        topic
        for topic, info in topic_info.items()
        if getattr(info, "msg_type", None) == POINTCLOUD2_TYPE
    )


def get_field(msg, name):
    for field in msg.fields:
        if field.name == name:
            return field
    return None


def field_dtype(field, is_bigendian):
    if field.count != 1:
        raise ValueError("field '{}' has count {}, expected 1".format(field.name, field.count))

    endian = ">" if is_bigendian else "<"
    if field.datatype == PointField.FLOAT32:
        return np.dtype(endian + "f4")
    if field.datatype == PointField.FLOAT64:
        return np.dtype(endian + "f8")

    raise ValueError(
        "field '{}' datatype {} is not FLOAT32/FLOAT64".format(field.name, field.datatype)
    )


def field_view(data, msg, field, dtype):
    height = msg.height
    width = msg.width
    if height == 0 or width == 0:
        return np.ndarray((0, 0), dtype=dtype)

    required_size = field.offset + (height - 1) * msg.row_step + (width - 1) * msg.point_step + dtype.itemsize
    if required_size > len(data):
        raise ValueError(
            "PointCloud2 data is too short for field '{}': need {}, got {}".format(
                field.name, required_size, len(data)
            )
        )

    return np.ndarray(
        shape=(height, width),
        dtype=dtype,
        buffer=data,
        offset=field.offset,
        strides=(msg.row_step, msg.point_step),
    )


def transform_pointcloud2(msg, rotation, translation, frame_id):
    out = copy.deepcopy(msg)
    out.header.frame_id = frame_id

    if msg.width == 0 or msg.height == 0:
        return out, 0

    x_field = get_field(msg, "x")
    y_field = get_field(msg, "y")
    z_field = get_field(msg, "z")
    if x_field is None or y_field is None or z_field is None:
        raise ValueError("PointCloud2 message does not contain x/y/z fields")

    data = bytearray(msg.data)
    x_dtype = field_dtype(x_field, msg.is_bigendian)
    y_dtype = field_dtype(y_field, msg.is_bigendian)
    z_dtype = field_dtype(z_field, msg.is_bigendian)

    x = field_view(data, msg, x_field, x_dtype)
    y = field_view(data, msg, y_field, y_dtype)
    z = field_view(data, msg, z_field, z_dtype)

    x_src = x.astype(np.float64)
    y_src = y.astype(np.float64)
    z_src = z.astype(np.float64)

    finite = np.isfinite(x_src) & np.isfinite(y_src) & np.isfinite(z_src)
    if np.any(finite):
        x_new = x_src.copy()
        y_new = y_src.copy()
        z_new = z_src.copy()

        xf = x_src[finite]
        yf = y_src[finite]
        zf = z_src[finite]

        x_new[finite] = (
            rotation[0, 0] * xf
            + rotation[0, 1] * yf
            + rotation[0, 2] * zf
            + translation[0]
        )
        y_new[finite] = (
            rotation[1, 0] * xf
            + rotation[1, 1] * yf
            + rotation[1, 2] * zf
            + translation[1]
        )
        z_new[finite] = (
            rotation[2, 0] * xf
            + rotation[2, 1] * yf
            + rotation[2, 2] * zf
            + translation[2]
        )

        x[...] = x_new.astype(x_dtype)
        y[...] = y_new.astype(y_dtype)
        z[...] = z_new.astype(z_dtype)

    out.data = bytes(data)
    return out, int(np.count_nonzero(finite))


def output_path_for(input_path, output_dir):
    stem = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.abspath(os.path.join(output_dir, stem + "_base_link.bag"))


def process_bag(input_path, output_path, topics, output_topic, rotation, translation, frame_id, copy_other_topics):
    transformed_msgs = 0
    transformed_points = 0

    with rosbag.Bag(input_path, "r") as in_bag:
        cloud_topics = get_pointcloud2_topics(in_bag)
        selected_topics = set(topics if topics is not None else cloud_topics)

        missing_topics = sorted(selected_topics.difference(cloud_topics))
        if missing_topics:
            print(
                "[WARN] {} does not contain PointCloud2 topic(s): {}".format(
                    input_path, ", ".join(missing_topics)
                ),
                file=sys.stderr,
            )

        selected_topics = selected_topics.intersection(cloud_topics)
        if not selected_topics:
            raise RuntimeError("no PointCloud2 topics selected in {}".format(input_path))

        if output_topic is not None and len(selected_topics) > 1:
            raise RuntimeError("--output-topic can only be used when one PointCloud2 topic is selected")

        with rosbag.Bag(output_path, "w") as out_bag:
            for topic, msg, stamp in in_bag.read_messages():
                if msg._type == POINTCLOUD2_TYPE and topic in selected_topics:
                    transformed_msg, valid_points = transform_pointcloud2(
                        msg, rotation, translation, frame_id
                    )
                    out_bag.write(output_topic or topic, transformed_msg, stamp)
                    transformed_msgs += 1
                    transformed_points += valid_points
                elif copy_other_topics:
                    out_bag.write(topic, msg, stamp)

    return transformed_msgs, transformed_points


def main():
    args = parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir or input_dir)
    os.makedirs(output_dir, exist_ok=True)

    rotation = np.array(args.rotation, dtype=np.float64).reshape(3, 3)
    translation = np.array(args.translation, dtype=np.float64)

    print("[Config] input_dir: {}".format(input_dir))
    print("[Config] output_dir: {}".format(output_dir))
    print("[Config] frame_id: {}".format(args.frame_id))
    print("[Config] extrinsic_T: {}".format(translation.tolist()))
    print("[Config] extrinsic_R:\n{}".format(rotation))

    outputs = []
    for bag_name in args.bags:
        input_path = resolve_bag_path(input_dir, bag_name)
        output_path = output_path_for(input_path, output_dir)

        if os.path.abspath(input_path) == os.path.abspath(output_path):
            raise RuntimeError("output path would overwrite input bag: {}".format(input_path))

        print("[Bag] {} -> {}".format(input_path, output_path))
        msg_count, point_count = process_bag(
            input_path=input_path,
            output_path=output_path,
            topics=args.topics,
            output_topic=args.output_topic,
            rotation=rotation,
            translation=translation,
            frame_id=args.frame_id,
            copy_other_topics=args.copy_other_topics,
        )
        print(
            "[Done] wrote {} PointCloud2 msg(s), transformed {} finite point(s)".format(
                msg_count, point_count
            )
        )
        outputs.append(output_path)

    print("[Output]")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
