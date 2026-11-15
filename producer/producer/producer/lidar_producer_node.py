"""LiDAR source: reads one LiDAR frame CSV and publishes it as a latched, organized sensor_msgs/PointCloud2.

Every pixel is published, including those without a return (distance 0), with every numeric column of the
file, so the consumer decides what to filter. Lengths are converted to metres (`unit_scale`).
"""
import numpy as np
from builtin_interfaces.msg import Time
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from producer.common import LATCHED, SourceNode, run

# CSV column -> (cloud field, numpy type, is a length that gets unit_scale)
COLUMNS = {
    "x": ("x", "f4", True), "y": ("y", "f4", True), "z": ("z", "f4", True),
    "Distance": ("distance", "f4", True), "Reflectivity": ("reflectivity", "f4", False),
    "Probability of False Alarm": ("pfa", "f4", False),
    "theta": ("theta", "f4", False), "phi": ("phi", "f4", False), "Pixel Number": ("pixel", "u4", False),
}
POINT_FIELD_TYPES = {"f4": PointField.FLOAT32, "u4": PointField.UINT32}


def load_frame(path, unit_scale):
    """Return (structured array of points, frame number, epoch time in microseconds)."""
    with open(path) as f:
        header = f.readline().rstrip("\n").split(",")
    numeric = [i for i, name in enumerate(header) if name != "Timestamp"]  # Timestamp = text copy of epoch
    raw = np.loadtxt(path, delimiter=",", skiprows=1, usecols=numeric)
    col = dict(zip((header[i] for i in numeric), raw.T))
    points = np.empty(len(raw), dtype=[(f, t) for f, t, _ in COLUMNS.values()])
    for name, (field, _, is_length) in COLUMNS.items():
        points[field] = col[name] * unit_scale if is_length else col[name]
    return points, int(col["Frame"][0]), int(col["Epoch time (usec)"][0])


def to_cloud(points, width, epoch_us, frame_id):
    fields = [PointField(name=name, offset=points.dtype.fields[name][1], count=1,
                         datatype=POINT_FIELD_TYPES[points.dtype.fields[name][0].str[1:]])
              for name in points.dtype.names]
    header = Header(frame_id=frame_id, stamp=Time(sec=epoch_us // 10**6, nanosec=(epoch_us % 10**6) * 1000))
    cloud = point_cloud2.create_cloud(header, fields, points)
    cloud.height, cloud.width = len(points) // width, width  # organized: keeps the sensor's pixel grid
    cloud.row_step = cloud.point_step * width
    cloud.is_dense = False
    return cloud


class LidarProducer(SourceNode):
    def __init__(self):
        super().__init__("lidar_producer", "csv_path", "unit_scale", "scan_width", "frame_id")
        points, frame, epoch_us = load_frame(self.cfg["csv_path"], self.cfg["unit_scale"])
        cloud = to_cloud(points, self.cfg["scan_width"], epoch_us, self.cfg["frame_id"])
        self.get_logger().info(f"loaded {self.cfg['csv_path']}: frame {frame}, {cloud.height}x{cloud.width} pixels, "
                               f"{np.count_nonzero(points['distance'] > 0)} returns, {len(cloud.data) / 1e6:.1f} MB")
        self.pub = self.create_publisher(PointCloud2, "/lidar/points", LATCHED)
        self.pub.publish(cloud)


def main():
    run(LidarProducer)
