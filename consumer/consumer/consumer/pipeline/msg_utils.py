"""Shared plumbing for pipeline stages: QoS, numpy <-> ROS message conversion, and a one-shot Stage node."""
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

# Latched: late subscribers still get the last message, so node start order never matters.
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)


def path_to_xy(msg: Path) -> np.ndarray:
    return np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses])


def xy_to_path(xy: np.ndarray, stamp, frame="map") -> Path:
    msg = Path()
    msg.header.frame_id, msg.header.stamp = frame, stamp
    for x, y in xy:
        p = PoseStamped()
        p.header = msg.header
        p.pose.position.x, p.pose.position.y = float(x), float(y)
        p.pose.orientation.w = 1.0
        msg.poses.append(p)
    return msg


def array_to_msg(a: np.ndarray) -> Float64MultiArray:
    a = np.asarray(a, dtype=float)
    msg = Float64MultiArray()
    msg.layout.dim = [MultiArrayDimension(label=f"d{i}", size=s) for i, s in enumerate(a.shape)]
    msg.data = a.ravel().tolist()
    return msg


def msg_to_array(msg: Float64MultiArray) -> np.ndarray:
    return np.array(msg.data).reshape([d.size for d in msg.layout.dim])


class Stage(Node):
    """Subscribes to `inputs` {name: (type, topic)}; calls `process(**msgs)` once, when all have arrived.

    Upstream topics are latched and republished, so processing once is enough; later messages are ignored.
    """

    def __init__(self, name, inputs):
        super().__init__(name)
        self.msgs, self.processed, self.finished = {}, False, False
        for key, (mtype, topic) in inputs.items():
            self.create_subscription(mtype, topic, lambda m, k=key: self._on_msg(k, m), LATCHED)
        self.n_inputs = len(inputs)

    def _on_msg(self, key, msg):
        self.msgs[key] = msg
        if not self.processed and len(self.msgs) == self.n_inputs:
            self.processed = True
            self.process(**self.msgs)

    def process(self, **msgs):
        raise NotImplementedError


def run(node_cls):
    """Spin a stage until it sets `finished` (report) or is interrupted (other stages)."""
    rclpy.init()
    node = node_cls()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
