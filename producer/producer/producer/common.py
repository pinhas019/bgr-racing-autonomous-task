"""Shared by the producer's source nodes: latched QoS, config parameters, and the source-node lifecycle."""
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
POLL_S = 0.5
STATUS_TOPIC = "/pipeline/status"


class SourceNode(Node):
    """Loads a dataset once, publishes it latched, and keeps serving it until the consumer publishes
    /pipeline/status (pipeline finished) or deadline_s passes. A crash is restarted by the launch file;
    the restarted node re-publishes the same latched data."""

    def __init__(self, name, *param_names):
        super().__init__(name)
        self.cfg = {}
        for n in ("deadline_s", *param_names):
            value = self.declare_parameter(n, descriptor=ParameterDescriptor(dynamic_typing=True)).value
            if value is None:
                raise RuntimeError(f"{name}: parameter '{n}' missing from config")
            self.cfg[n] = value
        self.finished = False
        self.create_subscription(String, STATUS_TOPIC, self.on_status, LATCHED)
        self.create_timer(self.cfg["deadline_s"], self.on_deadline)

    def on_status(self, msg):
        self.get_logger().info(f"pipeline status '{msg.data}'; exiting")
        self.finished = True

    def on_deadline(self):
        self.get_logger().error("deadline reached without a pipeline status; exiting")
        self.finished = True


def run(node_cls):
    rclpy.init()
    node = node_cls()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=POLL_S)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
