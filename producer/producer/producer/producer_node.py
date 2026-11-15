"""Producer: reads the track-boundary CSV and publishes each side as a latched nav_msgs/Path.

Keeps serving the latched data until the consumer publishes /pipeline/status (pipeline finished),
or until deadline_s passes. If it crashes, docker restarts it and it re-publishes the same latched data.
"""
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

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
        csv_path, deadline_s = (self.declare_parameter(n, descriptor=ParameterDescriptor(dynamic_typing=True)).value
                                for n in ("csv_path", "deadline_s"))
        if csv_path is None or deadline_s is None:
            raise RuntimeError("parameters 'csv_path' and 'deadline_s' must be set in the config")
        self.sides = load_boundaries(csv_path)
        self.get_logger().info(f"loaded {csv_path}: " + ", ".join(f"{s}={len(p)} points" for s, p in self.sides.items()))
        stamp = self.get_clock().now().to_msg()
        self.pubs = [self.create_publisher(Path, f"/track/{side}_boundary", LATCHED) for side in self.sides]
        for pub, pts in zip(self.pubs, self.sides.values()):
            pub.publish(to_path(pts, stamp))  # latched: delivered to subscribers whenever they (re)join
        self.finished = False
        self.create_subscription(String, "/pipeline/status", self.on_status, LATCHED)
        self.create_timer(deadline_s, self.on_deadline)

    def on_status(self, msg):
        self.get_logger().info(f"pipeline status '{msg.data}'; exiting")
        self.finished = True

    def on_deadline(self):
        self.get_logger().error("deadline reached without a pipeline status; exiting")
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
