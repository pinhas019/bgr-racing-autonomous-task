"""Consumer pipeline: boundaries -> centerline -> raceline -> speed profile -> report.

The report node exits after writing results; that shuts the whole launch down so the container exits.
"""
from launch import LaunchDescription
from launch.actions import EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node

VEHICLE = {"a_lat": 15.0, "a_acc": 8.0, "a_brk": 12.0, "v_top": 30.0}  # m/s^2, m/s


def generate_launch_description():
    report = Node(package="consumer", executable="report_node", output="screen",
                  parameters=[{"output_dir": "/output"}])
    return LaunchDescription([
        Node(package="consumer", executable="centerline_node", output="screen",
             parameters=[{"n_samples": 500}]),
        Node(package="consumer", executable="raceline_node", output="screen",
             parameters=[{"margin": 1.0}]),
        Node(package="consumer", executable="speed_profile_node", output="screen",
             parameters=[VEHICLE]),
        report,
        RegisterEventHandler(OnProcessExit(target_action=report,
                                           on_exit=[EmitEvent(event=Shutdown(reason="report done"))])),
    ])
