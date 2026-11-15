"""Stage 4: collect every pipeline output, write plots + results.json to output_dir, then exit."""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from nav_msgs.msg import Path
from std_msgs.msg import Float64MultiArray

from consumer.pipeline.centerline_node import left_normals
from consumer.pipeline.msg_utils import Stage, msg_to_array, path_to_xy, require_params, run
from consumer.pipeline.raceline_node import curvature, segment_lengths
from consumer.pipeline.speed_profile_node import lap_time

KMH = 3.6


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


class ReportNode(Stage):
    def __init__(self):
        super().__init__("report_node", {
            "left": (Path, "/track/left_boundary"), "right": (Path, "/track/right_boundary"),
            "center": (Path, "/track/centerline"), "width": (Float64MultiArray, "/track/width"),
            "race": (Path, "/track/raceline"),
            "v_center": (Float64MultiArray, "/speed/centerline"), "v_race": (Float64MultiArray, "/speed/raceline")})
        self.cfg = require_params(self, "output_dir", "timeout_s", "track_name")
        self.out = self.cfg["output_dir"]
        os.makedirs(self.out, exist_ok=True)
        self.results_file = os.path.join(self.out, "results.json")
        if os.path.exists(self.results_file):
            os.remove(self.results_file)  # a stale file must never look like a successful run
        self.create_timer(self.cfg["timeout_s"], self.on_timeout)

    def on_timeout(self):
        missing = sorted(self.input_keys - set(self.msgs))
        self.get_logger().error(f"timed out waiting for {missing}; no results written")
        self.finished = True

    def process(self, left, right, center, width, race, v_center, v_race):
        data = dict(left=path_to_xy(left), right=path_to_xy(right), center=path_to_xy(center),
                    widths=msg_to_array(width), race=path_to_xy(race),
                    v_center=msg_to_array(v_center), v_race=msg_to_array(v_race))
        results = summarize(**data)
        plot_track(os.path.join(self.out, "track.png"), self.cfg["track_name"], data["left"], data["right"], data["center"],
                   data["race"], data["v_race"], results)
        plot_profiles(os.path.join(self.out, "profiles.png"), data["center"], data["widths"], data["race"],
                      data["v_center"], data["v_race"])
        with open(self.results_file, "w") as f:
            json.dump(results, f, indent=2)
        self.get_logger().info(f"wrote track.png, profiles.png, results.json to {self.out}\n"
                               + json.dumps(results, indent=2))
        self.finished = True


def main():
    run(ReportNode)
