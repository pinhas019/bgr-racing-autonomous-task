"""Pure-function tests for the pipeline stages on a synthetic circular track (no ROS graph is started).

Run: docker compose run --rm consumer python3 -m pytest -q /ws/tests
"""
import numpy as np

from consumer.pipeline.centerline_node import compute_centerline, signed_area
from consumer.pipeline.report_node import edge_clearance
from consumer.pipeline.raceline_node import curvature, min_curvature_raceline, segment_lengths
from consumer.pipeline.speed_profile_node import lap_time, speed_profile

R, HALF_W = 100.0, 5.0
N_DENSE = 5000


def circle(r, n, ccw=True):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xy = np.column_stack([r * np.cos(t), r * np.sin(t)])
    return xy if ccw else xy[::-1]


def circle_track():
    # Driving counter-clockwise: the left boundary is the inner circle. Right is given clockwise on purpose.
    return circle(R - HALF_W, 157), circle(R + HALF_W, 143, ccw=False)


def test_centerline_on_circle():
    center, widths = compute_centerline(*circle_track(), n=400, n_dense=N_DENSE)
    assert center.shape == (400, 2) and widths.shape == (400, 2)
    assert np.allclose(np.linalg.norm(center, axis=1), R, atol=0.05)
    assert np.allclose(widths, HALF_W, atol=0.05)
    assert signed_area(center) > 0  # oriented in driving direction (CCW)
    spacing = np.linalg.norm(np.diff(center, axis=0), axis=1)
    assert spacing.std() / spacing.mean() < 0.01  # evenly resampled by arc length


MARGIN = 1.0
CAR = {"a_lat": 15.0, "a_acc": 8.0, "a_brk": 12.0, "v_top": 30.0}


def test_raceline_on_circle_hugs_outside():
    # Min curvature on a constant-radius corner = largest radius = outer edge (right side when driving CCW).
    center, widths = compute_centerline(*circle_track(), n=200, n_dense=N_DENSE)
    race, alpha = min_curvature_raceline(center, widths, MARGIN)
    assert np.allclose(np.linalg.norm(race, axis=1), R + HALF_W - MARGIN, atol=0.05)
    assert np.all(alpha >= -(widths[:, 1] - MARGIN) - 1e-9) and np.all(alpha <= widths[:, 0] - MARGIN + 1e-9)


def test_speed_profile_respects_limits():
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    xy = np.column_stack([300 * np.cos(t), 60 * np.sin(t)])  # ellipse: long straights + tight ends
    v = speed_profile(xy, **CAR)
    ds, k = segment_lengths(xy), np.abs(curvature(xy))
    dv2 = np.roll(v, -1) ** 2 - v ** 2
    assert np.all(v <= CAR["v_top"] + 1e-9)
    assert np.all(v ** 2 * k <= CAR["a_lat"] + 1e-6)          # cornering grip
    assert np.all(dv2 <= 2 * CAR["a_acc"] * ds + 1e-6)        # acceleration
    assert np.all(-dv2 <= 2 * CAR["a_brk"] * ds + 1e-6)       # braking
    assert v.min() < 0.8 * v.max()                              # it actually slows for the corners


def test_raceline_is_faster_than_centerline():
    # Ellipse-shaped track: the raceline should cut the corners and be quicker.
    t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    ell = lambda a, b: np.column_stack([a * np.cos(t), b * np.sin(t)])  # noqa: E731
    center, widths = compute_centerline(ell(200 - HALF_W, 60 - HALF_W), ell(200 + HALF_W, 60 + HALF_W), n=120, n_dense=N_DENSE)
    race, _ = min_curvature_raceline(center, widths, MARGIN)
    assert lap_time(race, speed_profile(race, **CAR)) < lap_time(center, speed_profile(center, **CAR))


def test_raceline_stays_inside_a_coarse_polygon_track():
    # Like the Brands Hatch data: few edge points joined by straight lines, with corners that poke inward.
    # The raceline must keep clear of the edges as drawn, also between its own samples.
    t = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    ell = lambda a, b: np.column_stack([a * np.cos(t), b * np.sin(t)])  # noqa: E731
    left, right = ell(200 - HALF_W, 60 - HALF_W), ell(200 + HALF_W, 60 + HALF_W)
    center, widths = compute_centerline(left, right, n=120, n_dense=N_DENSE)
    race, _ = min_curvature_raceline(center, widths, MARGIN)
    assert edge_clearance(race, left, right) > 0.8 * MARGIN
