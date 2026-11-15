"""Track source: reads the boundary CSV and publishes each side as a latched nav_msgs/Path."""
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

from producer.common import LATCHED, SourceNode, run


def load_boundaries(path):
    """Return {side: (N, 2) array}, skipping rows with empty coordinates (the dataset has one per side)."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    valid = ~(np.isnan(data["x"]) | np.isnan(data["y"]))
    return {side: np.column_stack([data["x"], data["y"]])[valid & (data["side"] == side)]
            for side in np.unique(data["side"])}


def to_path(points, stamp, frame):
    msg = Path()
    msg.header.frame_id, msg.header.stamp = frame, stamp
    for x, y in points:
        p = PoseStamped()
        p.header = msg.header
        p.pose.position.x, p.pose.position.y = float(x), float(y)
        p.pose.orientation.w = 1.0
        msg.poses.append(p)
    return msg


class TrackProducer(SourceNode):
    def __init__(self):
        super().__init__("track_producer", "csv_path", "frame_id")
        sides = load_boundaries(self.cfg["csv_path"])
        self.get_logger().info(f"loaded {self.cfg['csv_path']}: "
                               + ", ".join(f"{s}={len(p)} points" for s, p in sides.items()))
        stamp = self.get_clock().now().to_msg()
        self.pubs = {s: self.create_publisher(Path, f"/track/{s}_boundary", LATCHED) for s in sides}
        for side, pts in sides.items():
            self.pubs[side].publish(to_path(pts, stamp, self.cfg["frame_id"]))


def main():
    run(TrackProducer)
