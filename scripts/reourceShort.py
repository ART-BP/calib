#!/usr/bin/env python3
import rosbag

INPUT_BAG = "calib_data/scene_fd/left_f.bag"
OUTPUT_BAG = "calib_data/scene_fd/left_short_f.bag"
CLOUD_TOPIC = "/lidar_points"

START_INDEX = 10
NUM_FRAMES = 2

written = 0

with rosbag.Bag(INPUT_BAG, "r") as input_bag:
    with rosbag.Bag(OUTPUT_BAG, "w") as output_bag:
        for index, (topic, msg, t) in enumerate(
            input_bag.read_messages(topics=[CLOUD_TOPIC])
        ):
            if index < START_INDEX:
                continue

            if written >= NUM_FRAMES:
                break

            output_bag.write(topic, msg, t)
            written += 1

            print(
                f"write source frame {index}, "
                f"bag time = {t.to_sec():.6f}"
            )

print(f"finished: {written} point-cloud frames saved to {OUTPUT_BAG}")