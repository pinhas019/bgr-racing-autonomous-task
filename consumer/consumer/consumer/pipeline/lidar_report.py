"""LiDAR part of the report: summary statistics, plots and per-pixel / per-object tables (pure functions)."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from consumer.pipeline.lidar_cluster_node import COLUMNS
from consumer.pipeline.lidar_ground_node import LABELS, height_above

LABEL_COLORS = {"no_return": "#ffffff", "false_alarm": "#d62728", "ground": "#bdbdbd", "obstacle": "#1f77b4",
                "overhead": "#9467bd"}


def xyz_of(points):
    return np.column_stack([points["x"], points["y"], points["z"]]).astype(float)


def lidar_summary(points, labels, plane, table, stamp_s):
    returns = points["distance"] > 0
    cones = table[table[:, COLUMNS.index("is_cone")] == 1]
    col = lambda name: COLUMNS.index(name)  # noqa: E731
    return {
        "stamp_unix_s": stamp_s,
        "pixels": int(len(points)),
        "label_counts": {name: int(np.sum(labels == code)) for name, code in LABELS.items()},
        "return_range_m": {"min": round(float(points["distance"][returns].min()), 2),
                           "median": round(float(np.median(points["distance"][returns])), 2),
                           "max": round(float(points["distance"][returns].max()), 2)},
        "ground_plane": {"a": round(float(plane[0]), 5), "b": round(float(plane[1]), 5), "c": round(float(plane[2]), 4),
                         "tilt_deg": round(float(np.degrees(np.arctan(np.hypot(plane[0], plane[1])))), 2),
                         "sensor_height_m": round(float(-plane[2]), 3)},
        "objects": int(len(table)),
        "cone_candidates": [{"x": round(float(c[col("x")]), 2), "y": round(float(c[col("y")]), 2),
                             "range_m": round(float(c[col("range")]), 2),
                             "top_height_m": round(float(c[col("top_height")]), 2), "points": int(c[col("n_points")])}
                            for c in cones[np.argsort(cones[:, col("range")])]],
    }


def per_pixel_table(points, labels, plane, pixel_ids):
    """Every field of every pixel plus the pipeline's label, height above ground and object id."""
    columns = {name: points[name] for name in points.dtype.names}
    columns.update(label=labels, height_above_ground_m=height_above(xyz_of(points), plane), object_id=pixel_ids)
    return ",".join(columns), np.column_stack([np.asarray(v, dtype=float) for v in columns.values()])


def plot_bev(path, points, labels, pixel_ids, table, zoom_range):
    """Bird's-eye view: whole frame (left) and a zoom with the cone candidates annotated (right)."""
    xyz = xyz_of(points)
    ground, obstacle = labels == LABELS["ground"], labels == LABELS["obstacle"]
    cones = table[table[:, COLUMNS.index("is_cone")] == 1]
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    for ax, zoom in zip(axes, (False, True)):
        ax.scatter(xyz[ground, 0], xyz[ground, 1], s=0.2, c=LABEL_COLORS["ground"], rasterized=True)
        ax.scatter([], [], s=15, c=LABEL_COLORS["ground"], label="ground")  # legend-sized proxy
        ax.scatter(xyz[obstacle, 0], xyz[obstacle, 1], s=0.6, c=pixel_ids[obstacle] % 20, cmap="tab20",
                   rasterized=True)
        ax.scatter(cones[:, COLUMNS.index("x")], cones[:, COLUMNS.index("y")], s=120, facecolors="none",
                   edgecolors="orange", linewidths=2, label="cone candidate")
        ax.plot(0, 0, "k^", ms=10, label="sensor")
        ax.set_aspect("equal")
        ax.set_xlabel("x [m] (forward)")
        ax.set_ylabel("y [m] (left)")
        ax.grid(alpha=0.3)
        if zoom:
            ax.set_xlim(0, zoom_range)
            ax.set_ylim(-zoom_range / 1.5, zoom_range / 1.5)
            for c in cones:
                ax.annotate(f"{c[COLUMNS.index('top_height')]:.2f} m", (c[COLUMNS.index("x")], c[COLUMNS.index("y")]),
                            textcoords="offset points", xytext=(6, 6), fontsize=8)
            ax.set_title(f"zoom to {zoom_range:.0f} m: {len(cones)} cone candidates (label = top height)")
        else:
            ax.set_title(f"whole frame: {int(obstacle.sum())} obstacle points in {len(table)} objects "
                         "(colour = object)")
            ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def stitch_zones(image, zones):
    """(zones*lines, width) sensor layout -> (lines, zones*width) panorama; zones run left to right."""
    return np.hstack(np.split(image, zones, axis=0))


def plot_scan(path, points, labels, height, width, zones):
    """Range image and label image in the sensor's own pixel layout."""
    distance = stitch_zones(points["distance"].reshape(height, width), zones)
    label_img = stitch_zones(labels.reshape(height, width), zones)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(16, 7), sharex=True, layout="constrained")
    im = a1.imshow(np.where(distance > 0, distance, np.nan), aspect="auto", cmap="viridis", vmax=60)
    fig.colorbar(im, ax=[a1, a2], label="range [m] (clipped at 60)", shrink=0.5, anchor=(0, 1))
    a1.set_title("range image (white = no return)")
    cmap = ListedColormap([LABEL_COLORS[name] for name in sorted(LABELS, key=LABELS.get)])
    a2.imshow(label_img, aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(LABELS) - 0.5, interpolation="nearest")
    a2.set_title("pipeline labels: " + ", ".join(f"{n}" for n in sorted(LABELS, key=LABELS.get)
                                                 if n != "no_return") + " (white = no return)")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=LABEL_COLORS[n], ec="k") for n in sorted(LABELS, key=LABELS.get)]
    a2.legend(handles, sorted(LABELS, key=LABELS.get), loc="lower right", ncol=5, fontsize=8)
    for a in (a1, a2):
        for z in range(1, zones):
            a.axvline(z * width - 0.5, color="k", lw=0.8)
        a.set_ylabel("scan line")
    a2.set_xlabel(f"pixel column ({zones} zones of {width}, left -> right)")
    fig.savefig(path, dpi=110)
    plt.close(fig)
