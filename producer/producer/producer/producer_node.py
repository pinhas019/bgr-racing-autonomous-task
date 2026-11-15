"""Producer: reads the track-boundary CSV and publishes each side as a latched nav_msgs/Path.

Exits on its own once the consumer pipeline has received the data and finished (all subscribers gone).
"""
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
POLL_S = 0.5


def load_boundaries(path):
    """Return {side: (N, 2) array}, skipping rows with empty coordinates (the dataset has one per side)."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    valid = ~(np.isnan(data["x"]) | np.isnan(data["y"]))
    return {side: np.column_stack([data["x"], data["y"]])[valid & (data["side"] == side)]
            for side in np.unique(data["side"])}


def to_path(points, stamp, frame="map"):
    msg = Path()
    msg.header.frame_id, msg.header.stamp = frame, stamp
    for x, y in points:
        p = PoseStamped()
        p.header = msg.header
        p.pose.position.x, p.pose.position.y = float(x), float(y)
        p.pose.orientation.w = 1.0
        msg.poses.append(p)
    return msg


class Producer(Node):
    def __init__(self):
        super().__init__("producer")
        csv_path = self.declare_parameter("csv_path", descriptor=ParameterDescriptor(dynamic_typing=True)).value
        if csv_path is None:
            raise RuntimeError("parameter 'csv_path' missing from config")
        self.sides = load_boundaries(csv_path)
        self.get_logger().info(f"loaded {csv_path}: " + ", ".join(f"{s}={len(p)} points" for s, p in self.sides.items()))
        stamp = self.get_clock().now().to_msg()
        self.pubs = []
        for side, pts in self.sides.items():
            pub = self.create_publisher(Path, f"/track/{side}_boundary", LATCHED)
            pub.publish(to_path(pts, stamp))  # latched: delivered to subscribers whenever they join
            self.pubs.append(pub)
        self.had_subscribers = self.finished = False
        self.create_timer(POLL_S, self.check_done)

    def check_done(self):
        subscribers = sum(p.get_subscription_count() for p in self.pubs)
        self.had_subscribers |= subscribers > 0
        if self.had_subscribers and subscribers == 0:
            self.get_logger().info("consumer pipeline finished; exiting")
            self.finished = True


def main():
    rclpy.init()
    node = Producer()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=POLL_S)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
