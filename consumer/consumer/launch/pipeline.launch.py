"""Consumer pipeline: boundaries -> centerline -> raceline -> speed profile -> report.

Every stage exits on its own (see msg_utils.Stage); launch returns once all of them have finished.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

STAGES = ["centerline_node", "raceline_node", "speed_profile_node", "report_node"]


def generate_launch_description():
    config = os.path.join(get_package_share_directory("consumer"), "config", "pipeline.yaml")
    return LaunchDescription([
        Node(package="consumer", executable=stage, name=stage, output="screen", parameters=[config])
        for stage in STAGES
    ])
