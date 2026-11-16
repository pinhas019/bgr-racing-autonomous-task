"""Final stage: collect every pipeline output (track and LiDAR), write plots, tables and results.json, then tell
every node the pipeline is over via /pipeline/status."""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from nav_msgs.msg import Path
from rclpy.duration import Duration
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64MultiArray, Int32MultiArray, String

from consumer.pipeline.centerline_node import left_normals, signed_area
from consumer.pipeline.lidar_report import (lidar_summary, per_pixel_table, plot_bev, plot_cone_path, plot_scan,
                                            plot_scene)
from consumer.pipeline.msg_utils import (STATUS_TOPIC, Stage, cloud_to_array, msg_columns, msg_to_array, path_to_xy,
                                         require_params, run)
from consumer.pipeline.raceline_node import curvature, segment_lengths
from consumer.pipeline.speed_profile_node import lap_time

KMH = 3.6
OUTPUT_FILES = ("track_speed.png", "track_profiles.png", "track_data.csv", "cone_path.png", "cone_path_data.csv", "scene.png",
                "lidar_bev.png", "lidar_scan.png", "lidar_points.csv", "lidar_objects.csv", "results.json")


def edge_clearance(xy, left, right, closed=True, per_segment=20):
    """Smallest distance from the whole line (also between its samples) to the edges as given in the data
    (straight segments between the edge points). The line's samples lie inside the track, so a positive value
    means it never touches or crosses an edge."""
    pts = np.vstack([xy, xy[:1]]) if closed else xy
    f = np.linspace(0, 1, per_segment, endpoint=False)[:, None, None]
    line = (pts[:-1] + f * (pts[1:] - pts[:-1])).reshape(-1, 2)

    def distance(edge):
        a, b = (edge, np.roll(edge, -1, 0)) if closed else (edge[:-1], edge[1:])
        ab, ap = b - a, line[:, None] - a[None]
        t = np.clip(np.sum(ap * ab, axis=2) / np.sum(ab * ab, axis=1), 0, 1)
        return np.linalg.norm(ap - t[..., None] * ab, axis=2).min()

    return min(distance(left), distance(right))


def summarize(left, right, center, widths, race, v_center, v_race, closed=True):
    clearance = edge_clearance(race, left, right, closed)
    total_w = widths.sum(axis=1)
    t_c, t_r = lap_time(center, v_center, closed), lap_time(race, v_race, closed)
    return {
        "boundary_points": {"left": len(left), "right": len(right)},
        "centerline_length_m": round(float(segment_lengths(center, closed).sum()), 1),
        "raceline_length_m": round(float(segment_lengths(race, closed).sum()), 1),
        "track_width_m": {"min": round(float(total_w.min()), 2), "mean": round(float(total_w.mean()), 2),
                          "max": round(float(total_w.max()), 2)},
        "max_curvature_1_per_m": {"centerline": round(float(np.abs(curvature(center, closed)).max()), 4),
                                  "raceline": round(float(np.abs(curvature(race, closed)).max()), 4)},
        # closed: one lap; open: start of the path to its end
        ("lap_time_s" if closed else "time_s"): {"centerline": round(t_c, 2), "raceline": round(t_r, 2),
                                                 "gain": round(t_c - t_r, 2)},
        "speed_kmh": {"centerline": {"min": round(float(v_center.min() * KMH), 1), "max": round(float(v_center.max() * KMH), 1)},
                      "raceline": {"min": round(float(v_race.min() * KMH), 1), "max": round(float(v_race.max() * KMH), 1)}},
        "raceline_min_clearance_to_boundary_m": round(float(clearance), 2),
    }


def corner_minima(v_kmh, min_drop_kmh):
    """Indices of corners: local speed minima at least `min_drop_kmh` below the speeds around them (closed lap)."""
    n = len(v_kmh)
    peaks, _ = find_peaks(-np.tile(v_kmh, 3), prominence=min_drop_kmh)  # 3 laps so the start/finish wraps
    return np.unique(peaks[(peaks >= n) & (peaks < 2 * n)] - n)


def min_distance(xy, points, closed=True, per_segment=20):
    """Smallest distance from the whole line (also between its samples) to any of the points."""
    pts = np.vstack([xy, xy[:1]]) if closed else xy
    f = np.linspace(0, 1, per_segment, endpoint=False)[:, None, None]
    return float(cKDTree(points).query((pts[:-1] + f * (pts[1:] - pts[:-1])).reshape(-1, 2))[0].min())


def obstacle_summary(obstacles, pose, center, race):
    return {"assumed_lidar_pose": {"x": round(float(pose[0]), 2), "y": round(float(pose[1]), 2),
                                   "yaw_deg": round(float(np.degrees(pose[2])), 1)},
            "points": int(len(obstacles)), "objects": int(len(np.unique(obstacles[:, 2]))),
            # the raceline avoids them; the centerline is the plain reference and does not
            "raceline_min_distance_m": round(min_distance(race, obstacles[:, :2]), 2),
            "centerline_min_distance_m": round(min_distance(center, obstacles[:, :2]), 2)}


def draw_track(ax, left, right, xy, v_kmh, norm, obstacles, pose, lw):
    outer, inner = sorted((left, right), key=lambda b: -abs(signed_area(b)))
    ax.fill(*outer.T, color="0.88", lw=0)  # track surface = between the edges, exactly as in the data
    ax.fill(*inner.T, color="white", lw=0)
    for b in (left, right):
        ax.plot(*np.vstack([b, b[:1]]).T, color="0.2", lw=1.0)
    loop = np.vstack([xy, xy[:1]])
    lc = LineCollection(np.stack([loop[:-1], loop[1:]], axis=1), cmap="plasma", norm=norm, lw=lw, capstyle="round")
    lc.set_array(v_kmh)
    ax.add_collection(lc)
    ax.scatter(obstacles[:, 0], obstacles[:, 1], s=0.5, c="k", rasterized=True)
    ax.plot(*pose[:2], "ko", ms=5)
    ax.annotate("", xy=pose[:2] + 12 * np.array([np.cos(pose[2]), np.sin(pose[2])]), xytext=pose[:2],
                arrowprops={"arrowstyle": "->", "lw": 2, "color": "k"})
    ax.set_aspect("equal")
    return lc


def plot_track_speed(path, track_name, left, right, lines, results, corner_drop_kmh, obstacles, pose, zoom_pad=15.0):
    """One panel per line ({name: (xy, speed m/s)}), each coloured by its own speed on a shared scale,
    with the minimum speed written at every corner, the LiDAR obstacles placed on the map (black) and a zoom
    on them."""
    speeds = [v * KMH for _, v in lines.values()]
    norm = Normalize(min(v.min() for v in speeds), max(v.max() for v in speeds))
    lap = results["lap_time_s"]
    lo, hi = obstacles[:, :2].min(axis=0) - zoom_pad, obstacles[:, :2].max(axis=0) + zoom_pad
    fig, axes = plt.subplots(1, len(lines), figsize=(9 * len(lines), 10), layout="constrained")
    for ax, (name, (xy, _)), v_kmh in zip(axes, lines.items(), speeds):
        draw_track(ax, left, right, xy, v_kmh, norm, obstacles, pose, lw=2.5)
        zoom = ax.inset_axes([0.01, 0.01, 0.46, 0.4])
        draw_track(zoom, left, right, xy, v_kmh, norm, obstacles, pose, lw=4)
        zoom.set_xlim(lo[0], hi[0])
        zoom.set_ylim(lo[1], hi[1])
        zoom.set_xticks([])
        zoom.set_yticks([])
        d = results["obstacles"][f"{name}_min_distance_m"]
        zoom.set_title(f"LiDAR obstacles (black), closest {d:.1f} m", fontsize=9)
        ax.indicate_inset_zoom(zoom, edgecolor="k")
        for i in corner_minima(v_kmh, corner_drop_kmh):
            ax.annotate(f"{v_kmh[i]:.0f}", xy[i], textcoords="offset points", xytext=(8, 8), fontsize=9,
                        fontweight="bold", bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.8})
        ax.annotate("", xy=xy[4], xytext=xy[0], arrowprops={"arrowstyle": "->", "lw": 2.5})
        ax.plot(*xy[0], "ko", ms=7)
        ax.set_title(f"{name}: lap {lap[name]:.1f} s, {v_kmh.min():.0f}-{v_kmh.max():.0f} km/h", fontsize=13)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    fig.colorbar(ScalarMappable(norm, "plasma"), ax=axes, shrink=0.6, label="speed [km/h]")
    fig.suptitle(f"{track_name}: speed along the raceline vs the centerline (raceline -{lap['gain']:.1f} s)\n"
                 f"numbers = minimum speed in each corner [km/h], dot + arrow = start and driving direction\n"
                 f"black = LiDAR obstacles, placed with the LiDAR at an assumed pose (zoom: dot + arrow); "
                 f"only the raceline avoids them",
                 fontsize=14)
    fig.savefig(path, dpi=110)
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


def per_point_table(center, widths, race, v_center, v_race, closed=True):
    """Every per-point quantity the pipeline computed, one row per centerline sample."""
    columns = {
        "s_m": np.concatenate([[0], np.cumsum(segment_lengths(center, closed))[:-1]]),
        "center_x": center[:, 0], "center_y": center[:, 1],
        "width_left_m": widths[:, 0], "width_right_m": widths[:, 1],
        "race_x": race[:, 0], "race_y": race[:, 1],
        "race_offset_m": np.sum((race - center) * left_normals(center, closed), axis=1),
        "curvature_center": curvature(center, closed), "curvature_race": curvature(race, closed),
        "speed_center_mps": v_center, "speed_race_mps": v_race,
    }
    return ",".join(columns), np.column_stack(list(columns.values()))


class ReportNode(Stage):
    def __init__(self):
        super().__init__("report_node", {
            **planning_inputs("track"), **planning_inputs("cones"),
            "cloud": (PointCloud2, "/lidar/points"), "labels": (Int32MultiArray, "/lidar/labels"),
            "plane": (Float64MultiArray, "/lidar/ground_plane"), "objects": (Float64MultiArray, "/lidar/clusters"),
            "object_ids": (Int32MultiArray, "/lidar/cluster_ids"),
            "obstacles": (Float64MultiArray, "/track/obstacles"), "lidar_pose": (Float64MultiArray, "/track/lidar_pose")})
        self.cfg = require_params(self, "output_dir", "track_name", "ack_timeout_s", "scan_zones", "bev_zoom_range",
                                  "corner_drop_kmh")
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

    def report_track(self, m):
        data = planning_arrays(m, "track")
        obstacles, pose = msg_to_array(m["obstacles"]), msg_to_array(m["lidar_pose"])
        results = summarize(**data)
        results["obstacles"] = obstacle_summary(obstacles, pose, data["center"], data["race"])
        plot_track_speed(self.path("track_speed.png"), self.cfg["track_name"], data["left"], data["right"],
                         {"raceline": (data["race"], data["v_race"]), "centerline": (data["center"], data["v_center"])},
                         results, self.cfg["corner_drop_kmh"], obstacles, pose)
        plot_profiles(self.path("track_profiles.png"), data["center"], data["widths"], data["race"],
                      data["v_center"], data["v_race"])
        save_csv(self.path("track_data.csv"), *per_point_table(data["center"], data["widths"], data["race"],
                                                               data["v_center"], data["v_race"]))
        return results

    def report_cone_path(self, m):
        data = planning_arrays(m, "cones")
        results = summarize(**data, closed=False)
        points, labels, objects = cloud_to_array(m["cloud"]), msg_to_array(m["labels"]).ravel(), msg_to_array(m["objects"])
        plot_cone_path(self.path("cone_path.png"), points, labels, objects, data, results)
        plot_scene(self.path("scene.png"), points, labels, msg_to_array(m["object_ids"]).ravel(), objects, data,
                   results)
        save_csv(self.path("cone_path_data.csv"), *per_point_table(data["center"], data["widths"], data["race"],
                                                                   data["v_center"], data["v_race"], closed=False))
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

    def process(self, **m):
        results = {"track": self.report_track(m),
                   "lidar": self.report_lidar(m["cloud"], m["labels"], m["plane"], m["objects"], m["object_ids"]),
                   "cone_path": self.report_cone_path(m)}
        with open(self.path("results.json"), "w") as f:  # written last: its presence means the run succeeded
            json.dump(results, f, indent=2)
        self.get_logger().info(f"wrote {', '.join(OUTPUT_FILES)} to {self.out}\n" + json.dumps(results, indent=2))
        self.finish("success")


def planning_inputs(ns):
    """Report inputs from one planning namespace (/track or /cones), keyed '<ns>_<name>'."""
    return {f"{ns}_left": (Path, f"/{ns}/left_boundary"), f"{ns}_right": (Path, f"/{ns}/right_boundary"),
            f"{ns}_center": (Path, f"/{ns}/centerline"), f"{ns}_width": (Float64MultiArray, f"/{ns}/width"),
            f"{ns}_race": (Path, f"/{ns}/raceline"),
            f"{ns}_v_center": (Float64MultiArray, f"/{ns}/speed/centerline"),
            f"{ns}_v_race": (Float64MultiArray, f"/{ns}/speed/raceline")}


def planning_arrays(m, ns):
    return dict(left=path_to_xy(m[f"{ns}_left"]), right=path_to_xy(m[f"{ns}_right"]),
                center=path_to_xy(m[f"{ns}_center"]), widths=msg_to_array(m[f"{ns}_width"]),
                race=path_to_xy(m[f"{ns}_race"]), v_center=msg_to_array(m[f"{ns}_v_center"]),
                v_race=msg_to_array(m[f"{ns}_v_race"]))


def save_csv(path, header, table):
    np.savetxt(path, table, delimiter=",", header=header, comments="", fmt="%.6g")


def main():
    run(ReportNode)
