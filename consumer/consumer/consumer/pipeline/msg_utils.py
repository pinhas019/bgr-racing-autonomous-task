"""Shared plumbing for pipeline stages: QoS, config parameters, numpy <-> ROS messages, and the Stage node."""
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float64MultiArray, Int32MultiArray, MultiArrayDimension, String

# Latched: late subscribers still get the last message, so node start order never matters.
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
POLL_S = 0.5  # spin granularity
STATUS_TOPIC = "/pipeline/status"  # published once by the report: "success" or "failed: ..."


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


def array_to_msg(a, columns=None):
    """numpy array -> Float64MultiArray (Int32MultiArray for integer arrays). `columns` names the last axis."""
    a = np.asarray(a)
    msg = Int32MultiArray() if np.issubdtype(a.dtype, np.integer) else Float64MultiArray()
    msg.layout.dim = [MultiArrayDimension(label=f"d{i}", size=s) for i, s in enumerate(a.shape)]
    if columns is not None:
        msg.layout.dim[-1].label = ",".join(columns)
    msg.data = a.ravel().tolist()
    return msg


def msg_to_array(msg):
    return np.array(msg.data).reshape([d.size for d in msg.layout.dim])


def msg_columns(msg):
    """Column names stored by array_to_msg(..., columns=...)."""
    return msg.layout.dim[-1].label.split(",")


def cloud_to_array(msg: PointCloud2) -> np.ndarray:
    """PointCloud2 -> flat structured numpy array with every field (organized clouds keep row-major order)."""
    return point_cloud2.read_points(msg, skip_nans=False).ravel()


class Stage(Node):
    """One pipeline step. Subscribes to `inputs` {name: (type, topic)} and calls `process(**msgs)` once,
    when all have arrived. Publishers made with `self.output(...)` are latched.

    Fault handling:
    - Crash: the launch file restarts the node; latched inputs are re-delivered, so it simply recomputes.
    - Upstream lost before delivering: the node waits; past `deadline_s` it gives up and exits.
    - Finish: the node keeps serving its latched outputs until the report publishes /pipeline/status.
    """

    def __init__(self, name, inputs):
        super().__init__(name)
        self.msgs, self.input_keys = {}, set()
        self.processed = self.finished = False
        for key, (mtype, topic) in inputs.items():
            self.add_input(key, mtype, topic)
        self.create_subscription(String, STATUS_TOPIC, self.on_status, LATCHED)
        self.create_timer(require_params(self, "deadline_s")["deadline_s"], self.on_deadline)

    def add_input(self, key, mtype, topic):
        """An extra input, e.g. one that only some instances of a node need (set by a parameter)."""
        self.input_keys.add(key)
        self.create_subscription(mtype, topic, lambda m: self._on_msg(key, m), LATCHED)

    def output(self, mtype, topic):
        return self.create_publisher(mtype, topic, LATCHED)

    def _on_msg(self, key, msg):
        self.msgs[key] = msg
        if not self.processed and self.input_keys <= self.msgs.keys():
            self.processed = True
            self.process(**self.msgs)

    def on_status(self, msg):
        self.get_logger().info(f"pipeline status '{msg.data}'; exiting")
        self.finished = True

    def on_deadline(self):
        missing = sorted(self.input_keys - self.msgs.keys())
        self.get_logger().error(f"deadline reached (missing inputs: {missing}); exiting")
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
