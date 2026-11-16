"""Consumer pipeline.
  perception: point cloud -> lidar_ground -> lidar_cluster -> cone_track (cones -> left/right boundaries)
              obstacle_map: LiDAR obstacles placed on the map (assumed pose) -> /track raceline avoids them
  planning:   boundaries -> centerline -> raceline -> speed_profile, run twice with the SAME nodes:
              namespace /track  on the Brands Hatch map (closed lap)
              namespace /cones  on the boundaries built from LiDAR cones (open path)
  report:     everything -> output/

Each node is supervised: if it crashes (non-zero exit) it is started again after a delay, up to a retry limit.
A clean exit (pipeline finished) is never restarted. Settings: config/launch.yaml.
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import LogInfo, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node

PLANNING = ["centerline_node", "raceline_node", "speed_profile_node"]
PLANNING_NAMESPACES = ["track", "cones"]
OTHER = ["lidar_ground_node", "lidar_cluster_node", "cone_track_node", "obstacle_map_node", "report_node"]


def supervised(stage, namespace, params_file, retries_left, delay_s):
    node = Node(package="consumer", executable=stage, name=stage, namespace=namespace, output="screen",
                parameters=[params_file])
    label = f"{namespace}/{stage}" if namespace else stage

    def on_exit(event, _context):
        if event.returncode == 0:
            return None
        if retries_left == 0:
            return [LogInfo(msg=f"{label} failed (code {event.returncode}); no restarts left")]
        return [LogInfo(msg=f"{label} failed (code {event.returncode}); restarting in {delay_s} s, "
                            f"{retries_left} restart(s) left"),
                TimerAction(period=delay_s,
                            actions=supervised(stage, namespace, params_file, retries_left - 1, delay_s))]

    return [node, RegisterEventHandler(OnProcessExit(target_action=node, on_exit=on_exit))]


def generate_launch_description():
    share = get_package_share_directory("consumer")
    with open(os.path.join(share, "config", "launch.yaml")) as f:
        cfg = yaml.safe_load(f)
    params_file = os.path.join(share, "config", "pipeline.yaml")
    nodes = [(stage, ns) for ns in PLANNING_NAMESPACES for stage in PLANNING] + [(stage, "") for stage in OTHER]
    return LaunchDescription([action for stage, ns in nodes
                              for action in supervised(stage, ns, params_file, cfg["max_restarts"],
                                                       cfg["restart_delay_s"])])
