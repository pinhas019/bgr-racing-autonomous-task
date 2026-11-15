"""Pure-function tests: LiDAR cones -> boundaries, and the planner on an open path (no ROS graph is started)."""
import numpy as np

from consumer.pipeline.centerline_node import compute_centerline
from consumer.pipeline.cone_track_node import cone_boundaries
from consumer.pipeline.lidar_cluster_node import COLUMNS
from consumer.pipeline.raceline_node import min_curvature_raceline, segment_lengths
from consumer.pipeline.speed_profile_node import lap_time, speed_profile

CONE_TRACK = {"track_cone_min_height": 0.2, "track_cone_max_height": 0.4, "gate_min_width": 3.0,
              "gate_max_width": 7.0, "max_step": 16.0, "max_turn_deg": 60.0, "max_gate_skew_deg": 50.0}
CAR = {"a_lat": 15.0, "a_acc": 8.0, "a_brk": 12.0, "v_top": 30.0}
HALF_W, R = 2.5, 40.0


def bend(offset, n, angle=np.pi / 2):
    """Points on a right-hand bend starting 10 m ahead of the car: centre (10, -R), radius R + offset."""
    t = np.linspace(0, angle, n)
    return np.column_stack([10 + (R + offset) * np.sin(t), -R + (R + offset) * np.cos(t)])


def objects_table(rows):
    """rows of (x, y, top_height, is_cone) -> cluster table with COLUMNS."""
    table = np.zeros((len(rows), len(COLUMNS)))
    for k, (x, y, top, cone) in enumerate(rows):
        table[k, [COLUMNS.index(c) for c in ("id", "x", "y", "top_height", "is_cone")]] = (k, x, y, top, cone)
    return table


def test_cones_are_split_into_left_and_right():
    left, right = bend(HALF_W, 9), bend(-HALF_W, 9)  # left = outside of a right-hand bend
    rows = [(x, y, 0.32, 1) for x, y in np.vstack([left, right])]
    rows += [(25, 12, 0.65, 1)] * 1           # tall post: wrong cone type
    rows += [(30, -60, 0.32, 1)]              # small cone far off the track: never reached
    rows += [(15, 0, 1.5, 0)]                 # a wall segment: not a cone
    got_left, got_right, cones, gates = cone_boundaries(objects_table(rows), **CONE_TRACK)
    assert len(gates) >= 8
    assert np.allclose(got_left, left) and np.allclose(got_right, right)  # every cone, right side, in order


def test_open_path_plan_from_cones():
    left, right = bend(HALF_W, 9), bend(-HALF_W, 9)
    center, widths = compute_centerline(left, right, n=120, n_dense=1000, closed=False, interpolation="cubic")
    assert np.allclose(np.linalg.norm(center - [10, -R], axis=1), R, atol=0.1)  # midway between the edges
    assert center[0, 0] < center[-1, 0]  # runs away from the car
    race, alpha = min_curvature_raceline(center, widths, margin=1.0, closed=False, jacobian="dense")
    assert np.all(alpha <= widths[:, 0] - 1.0 + 1e-6) and np.all(alpha >= -(widths[:, 1] - 1.0) - 1e-6)
    v = speed_profile(race, **CAR, closed=False, v_start=0.0)
    ds = segment_lengths(race, closed=False)[:-1]
    assert v[0] == 0.0
    assert np.all(np.diff(v ** 2) <= 2 * CAR["a_acc"] * ds + 1e-6)  # standing start: acceleration-limited
    assert lap_time(race, v, closed=False) < lap_time(center, speed_profile(center, **CAR, closed=False, v_start=0.0),
                                                      closed=False) + 1e-9
