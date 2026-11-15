"""Shared plumbing for pipeline stages: QoS, config parameters, numpy <-> ROS messages, and the Stage node."""
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

# Latched: late subscribers still get the last message, so node start order never matters.
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
POLL_S = 0.5  # how often a finished stage checks whether its subscribers are gone


def require_params(node, *names):
    """Read parameters that must come from the YAML config: no defaults live in code."""
    values = {}
    for name in names:
        value = node.declare_parameter(name, descriptor=ParameterDescriptor(dynamic_typing=True)).value
        if value is None:
            raise RuntimeError(f"{node.get_name()}: parameter '{name}' missing from config")
        values[name] = value
    return values


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
    """One pipeline step. Subscribes to `inputs` {name: (type, topic)} and calls `process(**msgs)` once,
    when all have arrived. Publishers made with `self.output(...)` are latched.

    Lifecycle without signals: after processing, the stage keeps serving its latched outputs until every
    downstream subscriber has connected and gone again, then finishes. The last stage (report) finishes
    itself, so the whole pipeline unwinds from the end back to the producer.
    """

    def __init__(self, name, inputs):
        super().__init__(name)
        self.msgs, self.outputs, self.input_keys = {}, [], set(inputs)
        self.processed = self.had_subscribers = self.finished = False
        for key, (mtype, topic) in inputs.items():
            self.create_subscription(mtype, topic, lambda m, k=key: self._on_msg(k, m), LATCHED)
        self.create_timer(POLL_S, self._check_done)

    def output(self, mtype, topic):
        pub = self.create_publisher(mtype, topic, LATCHED)
        self.outputs.append(pub)
        return pub

    def _on_msg(self, key, msg):
        self.msgs[key] = msg
        if not self.processed and self.input_keys <= self.msgs.keys():
            self.processed = True
            self.process(**self.msgs)

    def _check_done(self):
        subscribers = sum(p.get_subscription_count() for p in self.outputs)
        self.had_subscribers |= subscribers > 0
        if self.outputs and self.had_subscribers and subscribers == 0:  # nobody left to serve
            self.get_logger().info("all downstream nodes are done; exiting")
            self.finished = True

    def process(self, **msgs):
        raise NotImplementedError


def run(node_cls):
    """Spin a stage until it finishes."""
    rclpy.init()
    node = node_cls()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=POLL_S)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
