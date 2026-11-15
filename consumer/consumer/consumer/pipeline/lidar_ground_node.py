"""LiDAR stage 1: label every pixel of the frame and estimate the ground plane.

1. Pixels without a return (distance 0) and returns with a high probability of false alarm are set aside.
2. Ground plane z = a*x + b*y + c is fitted to the returns within `fit_range` with a robust (Cauchy) loss
   (scipy least_squares), so walls, cones and cars barely pull on it.
3. Height above that plane splits the remaining returns into ground / obstacle / overhead.
"""
import numpy as np
from scipy.optimize import least_squares
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64MultiArray, Int32MultiArray

from consumer.pipeline.msg_utils import Stage, array_to_msg, cloud_to_array, require_params, run

LABELS = {"no_return": 0, "false_alarm": 1, "ground": 2, "obstacle": 3, "overhead": 4}


def fit_ground_plane(xyz, robust_scale):
    """Robust least-squares plane z = a x + b y + c. Returns (a, b, c)."""
    x, y, z = xyz.T
    start = [0.0, 0.0, np.percentile(z, 10)]  # ground is the low part of the scene
    res = least_squares(lambda p: z - (p[0] * x + p[1] * y + p[2]), start, loss="cauchy", f_scale=robust_scale)
    return res.x


def height_above(xyz, plane):
    a, b, c = plane
    return xyz[:, 2] - (a * xyz[:, 0] + b * xyz[:, 1] + c)


def label_points(points, max_pfa, fit_range, ground_tol, max_height, robust_scale):
    """Return (per-pixel label, ground plane (a, b, c), per-pixel height above the plane)."""
    xyz = np.column_stack([points["x"], points["y"], points["z"]]).astype(float)
    has_return = points["distance"] > 0
    trusted = has_return & (points["pfa"] <= max_pfa)
    plane = fit_ground_plane(xyz[trusted & (np.hypot(xyz[:, 0], xyz[:, 1]) < fit_range)], robust_scale)
    h = height_above(xyz, plane)
    labels = np.full(len(points), LABELS["no_return"], dtype=np.int32)
    labels[has_return & ~trusted] = LABELS["false_alarm"]
    labels[trusted & (h < ground_tol)] = LABELS["ground"]
    labels[trusted & (h >= ground_tol) & (h < max_height)] = LABELS["obstacle"]
    labels[trusted & (h >= max_height)] = LABELS["overhead"]
    return labels, plane, h


class LidarGroundNode(Stage):
    def __init__(self):
        super().__init__("lidar_ground_node", {"cloud": (PointCloud2, "/lidar/points")})
        self.cfg = require_params(self, "max_pfa", "fit_range", "ground_tol", "max_height", "robust_scale")
        self.pub_labels = self.output(Int32MultiArray, "/lidar/labels")
        self.pub_plane = self.output(Float64MultiArray, "/lidar/ground_plane")

    def process(self, cloud):
        labels, plane, _ = label_points(cloud_to_array(cloud), **self.cfg)
        counts = {name: int(np.sum(labels == code)) for name, code in LABELS.items()}
        tilt = np.degrees(np.arctan(np.hypot(plane[0], plane[1])))
        self.get_logger().info(f"ground plane z = {plane[0]:.4f}x + {plane[1]:.4f}y + {plane[2]:.3f} "
                               f"(tilt {tilt:.2f} deg); labels {counts}")
        self.pub_labels.publish(array_to_msg(labels.reshape(cloud.height, cloud.width)))
        self.pub_plane.publish(array_to_msg(plane, columns=["a", "b", "c"]))


def main():
    run(LidarGroundNode)
