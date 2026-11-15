"""Planning stage 1: track boundaries -> centerline + distances to each boundary.

Works for a closed lap (a race track map) and for an open path (a cone track seen by the LiDAR), set by the
`closed` parameter. Convention for everything downstream: the centerline runs in driving direction with the
LEFT boundary on its left, i.e. along the normal obtained by rotating the tangent +90 deg.
Widths are [dist_to_left, dist_to_right].
"""
import numpy as np
from nav_msgs.msg import Path
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree
from std_msgs.msg import Float64MultiArray

from consumer.pipeline.msg_utils import Stage, array_to_msg, path_to_xy, require_params, run, xy_to_path


def resample(xy, n, closed=True, interpolation="cubic"):
    """n points evenly spaced along a polyline (parameter = chord length, i.e. arc length of the polyline).
    interpolation "cubic": smooth spline through the points (periodic if closed, natural if open);
    "linear": straight segments between the points, e.g. a cone track, whose edges ARE straight lines."""
    xy = np.asarray(xy, float)
    if closed and np.allclose(xy[0], xy[-1]):
        xy = xy[:-1]
    pts = np.vstack([xy, xy[:1]]) if closed else xy
    u = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    s = np.linspace(0, u[-1], n, endpoint=not closed)
    if interpolation == "linear":
        return np.column_stack([np.interp(s, u, pts[:, 0]), np.interp(s, u, pts[:, 1])])
    return CubicSpline(u, pts, bc_type="periodic" if closed else "natural", axis=0)(s)


def left_normals(xy, closed=True):
    t = np.roll(xy, -1, 0) - np.roll(xy, 1, 0) if closed else np.gradient(xy, axis=0)
    t = t / np.linalg.norm(t, axis=1, keepdims=True)
    return np.column_stack([-t[:, 1], t[:, 0]])


def signed_area(xy):
    x, y = xy.T
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def same_direction(L, R, closed):
    if closed:
        return np.sign(signed_area(L)) == np.sign(signed_area(R))
    return np.dot(L[-1] - L[0], R[-1] - R[0]) > 0


def compute_centerline(left, right, n, n_dense, closed=True, interpolation="cubic"):
    """Midpoints between each left point and its nearest right point, re-splined and evenly resampled."""
    L, R = resample(left, n_dense, closed, interpolation), resample(right, n_dense, closed, interpolation)
    if not same_direction(L, R, closed):
        R = R[::-1]
    _, j = cKDTree(R).query(L)
    mid = (L + R[j]) / 2
    if not closed:  # drop the overhang where one boundary runs past the end of the other
        mid = mid[(j > 0) & (j < len(R) - 1)]
    center = resample(mid, n, closed)
    d_left, i_left = cKDTree(L).query(center)
    d_right, _ = cKDTree(R).query(center)
    # orient so the left boundary is on the +normal side (driving direction follows the side labels)
    if np.mean(np.sum((L[i_left] - center) * left_normals(center, closed), axis=1)) < 0:
        center, d_left, d_right = center[::-1], d_left[::-1], d_right[::-1]
    return center, np.column_stack([d_left, d_right])


class CenterlineNode(Stage):
    def __init__(self):
        super().__init__("centerline_node", {"left": (Path, "left_boundary"), "right": (Path, "right_boundary")})
        self.cfg = require_params(self, "n_samples", "n_dense", "closed", "interpolation")
        self.pub_center = self.output(Path, "centerline")
        self.pub_width = self.output(Float64MultiArray, "width")

    def process(self, left, right):
        center, widths = compute_centerline(path_to_xy(left), path_to_xy(right),
                                            self.cfg["n_samples"], self.cfg["n_dense"], self.cfg["closed"],
                                            self.cfg["interpolation"])
        total = widths.sum(axis=1)
        self.get_logger().info(f"centerline: {len(center)} pts, width min/mean/max "
                               f"{total.min():.1f}/{total.mean():.1f}/{total.max():.1f} m")
        self.pub_center.publish(xy_to_path(center, self.get_clock().now().to_msg()))
        self.pub_width.publish(array_to_msg(widths))


def main():
    run(CenterlineNode)
