"""Pure-function tests for the LiDAR stages on a synthetic scene (no ROS graph is started).

Scene: tilted ground plane with noise, one traffic cone, one wall, one pixel without a return and one
high-probability false alarm.
"""
import numpy as np

from consumer.pipeline.lidar_cluster_node import COLUMNS, cluster_obstacles
from consumer.pipeline.lidar_ground_node import LABELS, label_points

PLANE = (0.01, -0.02, -1.0)  # z = a x + b y + c
GROUND = {"max_pfa": 1e-3, "fit_range": 40.0, "ground_tol": 0.15, "max_height": 3.0, "robust_scale": 0.05}
CLUSTER = {"radius": 0.5, "min_points": 5, "cone_max_footprint": 0.5, "cone_min_height": 0.15, "cone_max_height": 0.8}


def on_plane(x, y, h):
    a, b, c = PLANE
    return a * x + b * y + c + h


def scene(seed=0):
    rng = np.random.default_rng(seed)
    gx, gy = rng.uniform(2, 30, 4000), rng.uniform(-15, 15, 4000)
    ground = np.column_stack([gx, gy, on_plane(gx, gy, rng.normal(0, 0.02, 4000))])
    # cone at (10, 3): 0.25 m wide, 0.45 m tall
    cz = rng.uniform(0.2, 0.45, 60)
    cone = np.column_stack([10 + rng.uniform(-0.12, 0.12, 60), 3 + rng.uniform(-0.12, 0.12, 60), on_plane(10, 3, cz)])
    # wall from (15, -10) to (15, 0): 10 m long, 1.5 m tall
    wy, wh = rng.uniform(-10, 0, 800), rng.uniform(0.2, 1.5, 800)
    wall = np.column_stack([np.full(800, 15.0), wy, on_plane(15, wy, wh)])
    xyz = np.vstack([ground, cone, wall, [[5, 5, 0]], [[20, 0, 5]]])
    points = np.zeros(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("distance", "f4"),
                                       ("reflectivity", "f4"), ("pfa", "f4")])
    points["x"], points["y"], points["z"] = xyz.T
    points["distance"] = np.linalg.norm(xyz, axis=1)
    points["pfa"] = 1e-4
    points["distance"][-2] = 0.0   # no return
    points["pfa"][-1] = 0.5        # false alarm
    return points, len(ground), len(cone), len(wall)


def test_ground_plane_and_labels():
    points, n_ground, n_cone, n_wall = scene()
    labels, plane, _ = label_points(points, **GROUND)
    assert np.allclose(plane, PLANE, atol=0.01)  # robust fit is not pulled up by the wall / cone
    assert labels[-2] == LABELS["no_return"] and labels[-1] == LABELS["false_alarm"]
    assert np.mean(labels[:n_ground] == LABELS["ground"]) > 0.99
    assert np.all(labels[n_ground:n_ground + n_cone + n_wall] == LABELS["obstacle"])


def test_clusters_find_cone_not_wall():
    points, n_ground, n_cone, n_wall = scene()
    labels, plane, _ = label_points(points, **GROUND)
    pixel_ids, table = cluster_obstacles(points, labels, plane, **CLUSTER)
    assert len(table) == 2
    cone = table[table[:, COLUMNS.index("is_cone")] == 1]
    assert len(cone) == 1
    assert np.allclose(cone[0, [COLUMNS.index("x"), COLUMNS.index("y")]], [10, 3], atol=0.1)
    assert 0.4 < cone[0, COLUMNS.index("top_height")] < 0.5
    assert np.all(pixel_ids[labels != LABELS["obstacle"]] == -1)
    assert len(np.unique(pixel_ids[n_ground:n_ground + n_cone])) == 1  # the cone is a single object
