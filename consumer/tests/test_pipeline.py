"""Pure-function tests for the pipeline stages on a synthetic circular track (no ROS graph is started).

Run: docker compose run --rm consumer python3 -m pytest -q /ws/tests
"""
import numpy as np

from consumer.pipeline.centerline_node import compute_centerline, signed_area

R, HALF_W = 100.0, 5.0


def circle(r, n, ccw=True):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xy = np.column_stack([r * np.cos(t), r * np.sin(t)])
    return xy if ccw else xy[::-1]


def circle_track():
    # Driving counter-clockwise: the left boundary is the inner circle. Right is given clockwise on purpose.
    return circle(R - HALF_W, 157), circle(R + HALF_W, 143, ccw=False)


def test_centerline_on_circle():
    center, widths = compute_centerline(*circle_track(), n=400)
    assert center.shape == (400, 2) and widths.shape == (400, 2)
    assert np.allclose(np.linalg.norm(center, axis=1), R, atol=0.05)
    assert np.allclose(widths, HALF_W, atol=0.05)
    assert signed_area(center) > 0  # oriented in driving direction (CCW)
    spacing = np.linalg.norm(np.diff(center, axis=0), axis=1)
    assert spacing.std() / spacing.mean() < 0.01  # evenly resampled by arc length
