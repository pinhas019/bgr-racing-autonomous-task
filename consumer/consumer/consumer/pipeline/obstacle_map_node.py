"""Perception -> map: the LiDAR obstacles placed on the track map, for the raceline to avoid.

One LiDAR frame carries no GPS pose, so where it was taken on the map is an ASSUMPTION set in the config: the car
stands on the centerline `anchor_s_m` metres after the start, facing along the track (LiDAR x = forward,
y = left). Only points labelled obstacle that belong to an object and lie within `max_range` of the sensor are
placed; the ground, false alarms and overhead structures are not obstacles for a car.
"""
import numpy as np
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64MultiArray, Int32MultiArray

from consumer.pipeline.lidar_ground_node import LABELS
from consumer.pipeline.msg_utils import (Stage, array_to_msg, cloud_to_array, msg_to_array, path_to_xy,
                                         require_params, run)
from consumer.pipeline.raceline_node import segment_lengths

COLUMNS = ["x", "y", "object_id"]


def anchor_pose(center, s):
    """(x, y, yaw) of the centerline point `s` metres after its start (closed lap), yaw along the track."""
    station = np.concatenate([[0.0], np.cumsum(segment_lengths(center))[:-1]])
    i = int(np.argmin(np.abs(station - s % station[-1])))
    tangent = np.roll(center, -1, 0)[i] - np.roll(center, 1, 0)[i]
    return np.array([*center[i], np.arctan2(tangent[1], tangent[0])])


def to_map(xy, pose):
    """Sensor-frame points -> map frame for a sensor at pose (x, y, yaw)."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return xy @ np.array([[c, s], [-s, c]]) + pose[:2]


def obstacle_points(points, labels, object_ids, max_range):
    """(x, y, object_id) in the sensor frame for every obstacle point of an object within range."""
    keep = (labels == LABELS["obstacle"]) & (object_ids >= 0) & (points["distance"] <= max_range)
    return np.column_stack([points["x"][keep], points["y"][keep], object_ids[keep]]).astype(float)


class ObstacleMapNode(Stage):
    def __init__(self):
        super().__init__("obstacle_map_node", {
            "cloud": (PointCloud2, "/lidar/points"), "labels": (Int32MultiArray, "/lidar/labels"),
            "object_ids": (Int32MultiArray, "/lidar/cluster_ids"), "center": (Path, "/track/centerline")})
        self.cfg = require_params(self, "anchor_s_m", "max_range")
        self.pub_obstacles = self.output(Float64MultiArray, "/track/obstacles")
        self.pub_pose = self.output(Float64MultiArray, "/track/lidar_pose")

    def process(self, cloud, labels, object_ids, center):
        pose = anchor_pose(path_to_xy(center), self.cfg["anchor_s_m"])
        obstacles = obstacle_points(cloud_to_array(cloud), msg_to_array(labels).ravel(),
                                    msg_to_array(object_ids).ravel(), self.cfg["max_range"])
        obstacles[:, :2] = to_map(obstacles[:, :2], pose)
        self.get_logger().info(f"LiDAR placed at map ({pose[0]:.1f}, {pose[1]:.1f}), heading "
                               f"{np.degrees(pose[2]):.1f} deg: {len(obstacles)} obstacle points of "
                               f"{len(np.unique(obstacles[:, 2]))} objects")
        self.pub_obstacles.publish(array_to_msg(obstacles, COLUMNS))
        self.pub_pose.publish(array_to_msg(pose, ["x", "y", "yaw"]))


def main():
    run(ObstacleMapNode)
