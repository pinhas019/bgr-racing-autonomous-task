"""Producer: reads the track-boundary CSV and publishes each side as a latched nav_msgs/Path."""
import csv

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)


def load_boundaries(path):
    """Return {'left': [(x, y), ...], 'right': [...]}, skipping rows with empty coordinates."""
    sides = {"left": [], "right": []}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["x"] and row["y"]:  # dataset has one empty row per side
                sides[row["side"]].append((float(row["x"]), float(row["y"])))
    return sides


def to_path(points, stamp, frame="map"):
    msg = Path()
    msg.header.frame_id = frame
    msg.header.stamp = stamp
    for x, y in points:
        p = PoseStamped()
        p.header = msg.header
        p.pose.position.x, p.pose.position.y = x, y
        p.pose.orientation.w = 1.0
        msg.poses.append(p)
    return msg


class Producer(Node):
    def __init__(self):
        super().__init__("producer")
        csv_path = self.declare_parameter("csv_path", "/data/BrandsHatchLayout.csv").value
        self.sides = load_boundaries(csv_path)
        self.get_logger().info(
            f"Loaded {csv_path}: left={len(self.sides['left'])} right={len(self.sides['right'])} points")
        self.pubs = {s: self.create_publisher(Path, f"/track/{s}_boundary", LATCHED) for s in self.sides}
        self.publish()
        self.create_timer(1.0, self.publish)  # republish for late joiners / visualisation

    def publish(self):
        stamp = self.get_clock().now().to_msg()
        for side, pts in self.sides.items():
            self.pubs[side].publish(to_path(pts, stamp))


def main():
    rclpy.init()
    node = Producer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
