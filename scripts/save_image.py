#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import threading

import cv2
import rospy
import rostopic
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage, Image


class ImageTopicSaver:
    def __init__(self, topic, output_dir, prefix, extension, encoding):
        self.topic = topic
        self.output_dir = os.path.abspath(output_dir)
        self.prefix = prefix
        self.extension = extension.lstrip(".").lower()
        self.encoding = encoding

        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_frame = None
        self.saved_count = 0

        os.makedirs(self.output_dir, exist_ok=True)

        msg_class, real_topic, _ = rostopic.get_topic_class(topic, blocking=True)
        if msg_class is None:
            raise RuntimeError("topic is not available: {}".format(topic))

        if msg_class == Image:
            self.subscriber = rospy.Subscriber(real_topic, Image, self.image_callback, queue_size=1)
        elif msg_class == CompressedImage:
            self.subscriber = rospy.Subscriber(
                real_topic, CompressedImage, self.compressed_image_callback, queue_size=1
            )
        else:
            raise RuntimeError(
                "unsupported topic type for {}: {}. Expected sensor_msgs/Image or "
                "sensor_msgs/CompressedImage".format(real_topic, msg_class._type)
            )

        self.topic = real_topic
        rospy.loginfo("subscribed to %s (%s)", self.topic, msg_class._type)
        if self.encoding != "passthrough":
            rospy.logwarn(
                "--encoding is set to %s. This may change pixel values; use "
                "'passthrough' to keep the original image pixels.",
                self.encoding,
            )
        rospy.loginfo("press Enter to save the latest image, or type q then Enter to quit")

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.encoding)
        except CvBridgeError as exc:
            rospy.logerr("failed to convert Image message: %s", exc)
            return

        image_encoding = msg.encoding if self.encoding == "passthrough" else self.encoding
        with self.lock:
            self.latest_frame = {
                "kind": "raw",
                "image": image.copy(),
                "encoding": image_encoding,
                "height": msg.height,
                "width": msg.width,
                "stamp": msg.header.stamp,
            }

    def compressed_image_callback(self, msg):
        with self.lock:
            self.latest_frame = {
                "kind": "compressed",
                "data": bytes(msg.data),
                "format": msg.format,
                "stamp": msg.header.stamp,
            }

    @staticmethod
    def compressed_extension(format_text, data):
        lower_format = (format_text or "").lower()
        if "jpeg" in lower_format or "jpg" in lower_format:
            return "jpg"
        if "png" in lower_format:
            return "png"

        if data.startswith(b"\xff\xd8\xff"):
            return "jpg"
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "png"
        if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            return "webp"
        return "bin"

    def extension_for_frame(self, frame):
        if self.extension != "auto":
            return self.extension
        if frame["kind"] == "compressed":
            return self.compressed_extension(frame["format"], frame["data"])
        return "png"

    @staticmethod
    def image_for_write(frame):
        image = frame["image"]
        height = frame["height"]
        width = frame["width"]
        if image.shape[:2] != (height, width):
            raise RuntimeError(
                "converted image size changed from {}x{} to {}x{}".format(
                    width, height, image.shape[1], image.shape[0]
                )
            )

        encoding = (frame["encoding"] or "").lower()
        if encoding in ("rgb8", "rgb16"):
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if encoding in ("rgba8", "rgba16"):
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        return image

    def save_latest(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            frame = self.latest_frame.copy()
            if frame["kind"] == "raw":
                frame["image"] = frame["image"].copy()

        stamp_text = "no_stamp"
        stamp = frame["stamp"]
        if stamp is not None and not stamp.is_zero():
            stamp_text = "{}_{:09d}".format(stamp.secs, stamp.nsecs)

        self.saved_count += 1
        extension = self.extension_for_frame(frame)
        filename = "{}_{:04d}_{}.{}".format(
            self.prefix, self.saved_count, stamp_text, extension
        )
        output_path = os.path.join(self.output_dir, filename)

        if frame["kind"] == "compressed":
            with open(output_path, "wb") as output_file:
                output_file.write(frame["data"])
        else:
            if extension in ("jpg", "jpeg"):
                raise RuntimeError(
                    "jpg/jpeg is lossy for raw Image topics; use png or tiff to keep pixels exact"
                )
            image = self.image_for_write(frame)
            if not cv2.imwrite(output_path, image):
                raise RuntimeError("failed to write image: {}".format(output_path))

        return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Subscribe to an image topic and save one image each time Enter is pressed."
    )
    parser.add_argument(
        "topic",
        nargs="?",
        default="/camera/image_raw",
        help="Image topic. Supports sensor_msgs/Image and sensor_msgs/CompressedImage.",
    )
    parser.add_argument(
        "--output-dir",
        default="images",
        help="Directory where images are saved. Defaults to ./images.",
    )
    parser.add_argument(
        "--prefix",
        default="image",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--extension",
        default="auto",
        help=(
            "Output image extension. Use auto to keep compressed image data in its "
            "original format and save raw Image topics as png."
        ),
    )
    parser.add_argument(
        "--encoding",
        default="passthrough",
        help=(
            "cv_bridge desired encoding for sensor_msgs/Image. Defaults to passthrough "
            "to keep original pixels."
        ),
    )
    return parser.parse_args(rospy.myargv()[1:])


def main():
    args = parse_args()
    rospy.init_node("image_topic_to_image", anonymous=True)
    saver = ImageTopicSaver(
        topic=args.topic,
        output_dir=args.output_dir,
        prefix=args.prefix,
        extension=args.extension,
        encoding=args.encoding,
    )

    while not rospy.is_shutdown():
        try:
            command = input()
        except EOFError:
            break
        except KeyboardInterrupt:
            break

        if command.strip().lower() in ("q", "quit", "exit"):
            break

        try:
            output_path = saver.save_latest()
        except RuntimeError as exc:
            rospy.logerr("%s", exc)
            continue

        if output_path is None:
            rospy.logwarn("no image received yet")
        else:
            rospy.loginfo("saved image: %s", output_path)


if __name__ == "__main__":
    main()
