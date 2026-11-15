"""Stage 1: track boundaries -> centerline + distances to each boundary.

Convention for everything downstream: the centerline runs in driving direction with the LEFT boundary on its
left, i.e. along the normal obtained by rotating the tangent +90 deg. Widths are [dist_to_left, dist_to_right].
"""
import numpy as np
from nav_msgs.msg import Path
from scipy.interpolate import splev, splprep
from scipy.spatial import cKDTree
from std_msgs.msg import Float64MultiArray

from consumer.pipeline.msg_utils import LATCHED, Stage, array_to_msg, path_to_xy, run, xy_to_path


def open_loop(xy):
    """Drop a duplicated closing point, if present."""
    return xy[:-1] if np.allclose(xy[0], xy[-1]) else xy


def resample_closed(xy, n):
    """Fit a periodic spline through a closed polyline and return n points equally spaced by arc length."""
    xy = open_loop(np.asarray(xy, float))
    loop = np.vstack([xy, xy[:1]])  # splprep(per=1) wants the first point repeated at the end
    tck, _ = splprep(loop.T, s=0, per=1)  # exact interpolation; smoothing only cost accuracy
    u = np.linspace(0, 1, 20 * n, endpoint=False)
    dense = np.column_stack(splev(u, tck))
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0, append=dense[:1]), axis=1))])
    u_even = np.interp(np.linspace(0, s[-1], n, endpoint=False), s, np.append(u, 1.0))
    return np.column_stack(splev(u_even, tck))


def left_normals(xy):
    t = np.roll(xy, -1, 0) - np.roll(xy, 1, 0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return np.column_stack([-t[:, 1], t[:, 0]])


def signed_area(xy):
    x, y = xy.T
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def compute_centerline(left, right, n=500, n_dense=5000):
    """Midpoints between each left point and its nearest right point, re-splined and evenly resampled."""
    L, R = resample_closed(left, n_dense), resample_closed(right, n_dense)
    if np.sign(signed_area(L)) != np.sign(signed_area(R)):
        R = R[::-1]  # make both boundaries run the same way round
    _, j = cKDTree(R).query(L)
    center = resample_closed((L + R[j]) / 2, n)
    d_left, i_left = cKDTree(L).query(center)
    d_right, _ = cKDTree(R).query(center)
    # orient so the left boundary is on the +normal side (driving direction follows the side labels)
    if np.mean(np.sum((L[i_left] - center) * left_normals(center), axis=1)) < 0:
        center, d_left, d_right = center[::-1], d_left[::-1], d_right[::-1]
    return center, np.column_stack([d_left, d_right])


class CenterlineNode(Stage):
    def __init__(self):
        super().__init__("centerline_node", {"left": (Path, "/track/left_boundary"),
                                             "right": (Path, "/track/right_boundary")})
        self.n = self.declare_parameter("n_samples", 500).value
        self.pub_center = self.create_publisher(Path, "/track/centerline", LATCHED)
        self.pub_width = self.create_publisher(Float64MultiArray, "/track/width", LATCHED)

    def process(self, left, right):
        center, widths = compute_centerline(path_to_xy(left), path_to_xy(right), self.n)
        total = widths.sum(axis=1)
        self.get_logger().info(f"centerline: {len(center)} pts, width min/mean/max "
                               f"{total.min():.1f}/{total.mean():.1f}/{total.max():.1f} m")
        self.pub_center.publish(xy_to_path(center, self.get_clock().now().to_msg()))
        self.pub_width.publish(array_to_msg(widths))


def main():
    run(CenterlineNode)
