"""Consumer pipeline.
  track: boundaries -> centerline -> raceline -> speed profile -> report
  LiDAR: point cloud -> ground/labels -> clusters/cones -> report

Each stage is supervised: if it crashes (non-zero exit) it is started again after a delay, up to a retry limit.
A clean exit (pipeline finished) is never restarted. Settings: config/launch.yaml.
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import LogInfo, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node

STAGES = ["centerline_node", "raceline_node", "speed_profile_node", "lidar_ground_node", "lidar_cluster_node",
          "report_node"]


def supervised(stage, params_file, retries_left, delay_s):
    node = Node(package="consumer", executable=stage, name=stage, output="screen", parameters=[params_file])

    def on_exit(event, _context):
        if event.returncode == 0:
            return None
        if retries_left == 0:
            return [LogInfo(msg=f"{stage} failed (code {event.returncode}); no restarts left")]
        return [LogInfo(msg=f"{stage} failed (code {event.returncode}); restarting in {delay_s} s, "
                            f"{retries_left} restart(s) left"),
                TimerAction(period=delay_s, actions=supervised(stage, params_file, retries_left - 1, delay_s))]

    return [node, RegisterEventHandler(OnProcessExit(target_action=node, on_exit=on_exit))]


def generate_launch_description():
    share = get_package_share_directory("consumer")
    with open(os.path.join(share, "config", "launch.yaml")) as f:
        cfg = yaml.safe_load(f)
    params_file = os.path.join(share, "config", "pipeline.yaml")
    return LaunchDescription([action for stage in STAGES
                              for action in supervised(stage, params_file, cfg["max_restarts"], cfg["restart_delay_s"])])
