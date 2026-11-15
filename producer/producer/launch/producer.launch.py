"""Producer: one source node per dataset (track boundaries, LiDAR frame).

Each node is supervised: if it crashes (non-zero exit) it is started again after a delay, up to a retry limit.
A clean exit (pipeline finished) is never restarted. Settings: config/launch.yaml.
(Same supervision as the consumer launch file; the two images deliberately share no code.)
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import LogInfo, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node

NODES = {"track_producer_node": "track_producer", "lidar_producer_node": "lidar_producer"}


def supervised(executable, name, params_file, retries_left, delay_s):
    node = Node(package="producer", executable=executable, name=name, output="screen", parameters=[params_file])

    def on_exit(event, _context):
        if event.returncode == 0:
            return None
        if retries_left == 0:
            return [LogInfo(msg=f"{name} failed (code {event.returncode}); no restarts left")]
        return [LogInfo(msg=f"{name} failed (code {event.returncode}); restarting in {delay_s} s, "
                            f"{retries_left} restart(s) left"),
                TimerAction(period=delay_s, actions=supervised(executable, name, params_file, retries_left - 1, delay_s))]

    return [node, RegisterEventHandler(OnProcessExit(target_action=node, on_exit=on_exit))]


def generate_launch_description():
    share = get_package_share_directory("producer")
    with open(os.path.join(share, "config", "launch.yaml")) as f:
        cfg = yaml.safe_load(f)
    params_file = os.path.join(share, "config", "producer.yaml")
    return LaunchDescription([action for exe, name in NODES.items()
                              for action in supervised(exe, name, params_file, cfg["max_restarts"], cfg["restart_delay_s"])])
