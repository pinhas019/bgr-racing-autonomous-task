"""Final stage: collect every pipeline output (track and LiDAR), write plots, tables and results.json, then tell
every node the pipeline is over via /pipeline/status."""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from nav_msgs.msg import Path
from rclpy.duration import Duration
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64MultiArray, Int32MultiArray, String

from consumer.pipeline.centerline_node import left_normals
from consumer.pipeline.lidar_report import lidar_summary, per_pixel_table, plot_bev, plot_scan
from consumer.pipeline.msg_utils import (STATUS_TOPIC, Stage, cloud_to_array, msg_columns, msg_to_array, path_to_xy,
                                         require_params, run)
from consumer.pipeline.raceline_node import curvature, segment_lengths
from consumer.pipeline.speed_profile_node import lap_time

KMH = 3.6
OUTPUT_FILES = ("track.png", "track_profiles.png", "track_data.csv",
                "lidar_bev.png", "lidar_scan.png", "lidar_points.csv", "lidar_objects.csv", "results.json")


def summarize(left, right, center, widths, race, v_center, v_race):
    offset = np.sum((race - center) * left_normals(center), axis=1)
    clearance = np.minimum(widths[:, 0] - offset, widths[:, 1] + offset).min()
    total_w = widths.sum(axis=1)
    t_c, t_r = lap_time(center, v_center), lap_time(race, v_race)
    return {
        "boundary_points": {"left": len(left), "right": len(right)},
        "centerline_length_m": round(float(segment_lengths(center).sum()), 1),
        "raceline_length_m": round(float(segment_lengths(race).sum()), 1),
        "track_width_m": {"min": round(float(total_w.min()), 2), "mean": round(float(total_w.mean()), 2),
                          "max": round(float(total_w.max()), 2)},
        "max_curvature_1_per_m": {"centerline": round(float(np.abs(curvature(center)).max()), 4),
                                  "raceline": round(float(np.abs(curvature(race)).max()), 4)},
        "lap_time_s": {"centerline": round(t_c, 2), "raceline": round(t_r, 2), "gain": round(t_c - t_r, 2)},
        "speed_kmh": {"centerline": {"min": round(float(v_center.min() * KMH), 1), "max": round(float(v_center.max() * KMH), 1)},
                      "raceline": {"min": round(float(v_race.min() * KMH), 1), "max": round(float(v_race.max() * KMH), 1)}},
        "raceline_min_clearance_to_boundary_m": round(float(clearance), 2),
    }


def plot_track(path, track_name, left, right, center, race, v_race, results):
    fig, ax = plt.subplots(figsize=(10, 10))
    for b in (left, right):
        loop = np.vstack([b, b[:1]])
        ax.plot(*loop.T, color="0.35", lw=1.2)
    ax.plot(*np.vstack([center, center[:1]]).T, "--", color="0.6", lw=0.8, label="centerline")
    loop = np.vstack([race, race[:1]])
    lc = LineCollection(np.stack([loop[:-1], loop[1:]], axis=1), cmap="plasma", lw=2.5)
    lc.set_array(v_race * KMH)
    ax.add_collection(lc)
    fig.colorbar(lc, ax=ax, shrink=0.6, label="raceline speed [km/h]")
    ax.annotate("", xy=race[3], xytext=race[0], arrowprops={"arrowstyle": "->", "lw": 2})
    ax.plot(*race[0], "ko", label="start / finish")
    lt = results["lap_time_s"]
    ax.set_title(f"{track_name}: minimum-curvature raceline\n"
                 f"lap {lt['raceline']:.1f} s vs centerline {lt['centerline']:.1f} s (-{lt['gain']:.1f} s)")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper right")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_profiles(path, center, widths, race, v_center, v_race):
    s = np.concatenate([[0], np.cumsum(segment_lengths(center))[:-1]])  # station along the centerline
    offset = np.sum((race - center) * left_normals(center), axis=1)
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    a1.fill_between(s, -widths[:, 1], widths[:, 0], color="0.9", label="track")
    a1.plot(s, offset, color="C1", label="raceline")
    a1.axhline(0, color="0.5", ls="--", lw=0.8, label="centerline")
    a1.set_ylabel("lateral offset [m]\n(+ = left)")
    a2.plot(s, np.abs(curvature(center)), color="0.5", label="centerline")
    a2.plot(s, np.abs(curvature(race)), color="C1", label="raceline")
    a2.set_ylabel("|curvature| [1/m]")
    a3.plot(s, v_center * KMH, color="0.5", label="centerline")
    a3.plot(s, v_race * KMH, color="C1", label="raceline")
    a3.set_ylabel("speed [km/h]")
    a3.set_xlabel("distance along centerline [m]")
    for a in (a1, a2, a3):
        a.legend(loc="upper right")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def per_point_table(center, widths, race, v_center, v_race):
    """Every per-point quantity the pipeline computed, one row per centerline sample."""
    columns = {
        "s_m": np.concatenate([[0], np.cumsum(segment_lengths(center))[:-1]]),
        "center_x": center[:, 0], "center_y": center[:, 1],
        "width_left_m": widths[:, 0], "width_right_m": widths[:, 1],
        "race_x": race[:, 0], "race_y": race[:, 1],
        "race_offset_m": np.sum((race - center) * left_normals(center), axis=1),
        "curvature_center": curvature(center), "curvature_race": curvature(race),
        "speed_center_mps": v_center, "speed_race_mps": v_race,
    }
    return ",".join(columns), np.column_stack(list(columns.values()))


class ReportNode(Stage):
    def __init__(self):
        super().__init__("report_node", {
            "left": (Path, "/track/left_boundary"), "right": (Path, "/track/right_boundary"),
            "center": (Path, "/track/centerline"), "width": (Float64MultiArray, "/track/width"),
            "race": (Path, "/track/raceline"),
            "v_center": (Float64MultiArray, "/speed/centerline"), "v_race": (Float64MultiArray, "/speed/raceline"),
            "cloud": (PointCloud2, "/lidar/points"), "labels": (Int32MultiArray, "/lidar/labels"),
            "plane": (Float64MultiArray, "/lidar/ground_plane"), "objects": (Float64MultiArray, "/lidar/clusters"),
            "object_ids": (Int32MultiArray, "/lidar/cluster_ids")})
        self.cfg = require_params(self, "output_dir", "track_name", "ack_timeout_s", "scan_zones", "bev_zoom_range")
        self.out = self.cfg["output_dir"]
        self.status_pub = self.output(String, STATUS_TOPIC)
        os.makedirs(self.out, exist_ok=True)
        for name in OUTPUT_FILES:  # stale outputs must never look like results of this run
            if os.path.exists(self.path(name)):
                os.remove(self.path(name))

    def path(self, name):
        return os.path.join(self.out, name)

    def finish(self, status):
        """Tell every node (both containers) the pipeline is over, and wait until they all got it."""
        self.status_pub.publish(String(data=status))
        if not self.status_pub.wait_for_all_acked(Duration(seconds=self.cfg["ack_timeout_s"])):
            self.get_logger().warning("not every node acknowledged the pipeline status")
        self.finished = True

    def on_status(self, msg):
        pass  # the report is the one publishing it

    def on_deadline(self):
        missing = sorted(self.input_keys - self.msgs.keys())
        self.get_logger().error(f"deadline reached waiting for {missing}; no results written")
        self.finish(f"failed: missing {missing}")

    def report_track(self, left, right, center, width, race, v_center, v_race):
        data = dict(left=path_to_xy(left), right=path_to_xy(right), center=path_to_xy(center),
                    widths=msg_to_array(width), race=path_to_xy(race),
                    v_center=msg_to_array(v_center), v_race=msg_to_array(v_race))
        results = summarize(**data)
        plot_track(self.path("track.png"), self.cfg["track_name"], data["left"], data["right"],
                   data["center"], data["race"], data["v_race"], results)
        plot_profiles(self.path("track_profiles.png"), data["center"], data["widths"], data["race"],
                      data["v_center"], data["v_race"])
        save_csv(self.path("track_data.csv"), *per_point_table(data["center"], data["widths"], data["race"],
                                                               data["v_center"], data["v_race"]))
        return results

    def report_lidar(self, cloud, labels, plane, objects, object_ids):
        points, labels = cloud_to_array(cloud), msg_to_array(labels).ravel()
        plane, table, ids = msg_to_array(plane), msg_to_array(objects), msg_to_array(object_ids).ravel()
        stamp = cloud.header.stamp.sec + cloud.header.stamp.nanosec * 1e-9
        plot_bev(self.path("lidar_bev.png"), points, labels, ids, table, self.cfg["bev_zoom_range"])
        plot_scan(self.path("lidar_scan.png"), points, labels, cloud.height, cloud.width, self.cfg["scan_zones"])
        save_csv(self.path("lidar_points.csv"), *per_pixel_table(points, labels, plane, ids))
        save_csv(self.path("lidar_objects.csv"), ",".join(msg_columns(objects)), table)
        return lidar_summary(points, labels, plane, table, stamp)

    def process(self, left, right, center, width, race, v_center, v_race, cloud, labels, plane, objects, object_ids):
        results = {"track": self.report_track(left, right, center, width, race, v_center, v_race),
                   "lidar": self.report_lidar(cloud, labels, plane, objects, object_ids)}
        with open(self.path("results.json"), "w") as f:  # written last: its presence means the run succeeded
            json.dump(results, f, indent=2)
        self.get_logger().info(f"wrote {', '.join(OUTPUT_FILES)} to {self.out}\n" + json.dumps(results, indent=2))
        self.finish("success")


def save_csv(path, header, table):
    np.savetxt(path, table, delimiter=",", header=header, comments="", fmt="%.6g")


def main():
    run(ReportNode)
