"""Stage 2: centerline + widths -> minimum-curvature raceline.

The raceline is the centerline shifted along its left normal by a lateral offset alpha_i, bounded so the car
(plus a safety margin) stays inside the track. We minimise the curvature energy  sum(kappa_i^2 * ds_i)
using the exact (Menger) curvature of the shifted points, written as least-squares residuals
r_i = kappa_i * sqrt(ds_i) and solved with bounded Gauss-Newton (scipy least_squares, 'trf').
r_i only depends on alpha_{i-1..i+1}, so the Jacobian is cyclic-tridiagonal and cheap to build.
"""
import numpy as np
from nav_msgs.msg import Path
from scipy.optimize import least_squares
from scipy.sparse import diags
from std_msgs.msg import Float64MultiArray

from consumer.pipeline.centerline_node import left_normals
from consumer.pipeline.msg_utils import Stage, msg_to_array, path_to_xy, require_params, run, xy_to_path


def curvature(xy):
    """Signed curvature of a closed polyline at each vertex: 1 / circumradius of (prev, this, next)."""
    prev, nxt = np.roll(xy, 1, 0), np.roll(xy, -1, 0)
    a, b, c = xy - prev, nxt - xy, nxt - prev
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return 2 * cross / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) * np.linalg.norm(c, axis=1))


def segment_lengths(xy):
    """ds[i] = distance from point i to point i+1 (closed loop)."""
    return np.linalg.norm(np.roll(xy, -1, 0) - xy, axis=1)


def curvature_energy(xy):
    return np.sum(curvature(xy) ** 2 * segment_lengths(xy))


def min_curvature_raceline(center, widths, margin):
    """Return (raceline xy, lateral offsets). widths[:, 0/1] = distance to left/right boundary."""
    n = len(center)
    normals = left_normals(center)
    hi = np.maximum(widths[:, 0] - margin, 1e-6)  # least_squares needs lo < hi strictly
    lo = np.minimum(-(widths[:, 1] - margin), -1e-6)

    def residuals(alpha):
        p = center + alpha[:, None] * normals
        return curvature(p) * np.sqrt(segment_lengths(p))

    # scipy builds the finite-difference Jacobian itself; the sparsity pattern lets it perturb
    # non-interacting columns together (a few evals per Jacobian instead of n).
    sparsity = diags([1, 1, 1, 1, 1], [-n + 1, -1, 0, 1, n - 1], shape=(n, n))  # cyclic tridiagonal
    res = least_squares(residuals, np.zeros(n), bounds=(lo, hi), jac_sparsity=sparsity, method="trf")
    return center + res.x[:, None] * normals, res.x


class RacelineNode(Stage):
    def __init__(self):
        super().__init__("raceline_node", {"center": (Path, "/track/centerline"),
                                           "width": (Float64MultiArray, "/track/width")})
        self.margin = require_params(self, "margin")["margin"]
        self.pub = self.output(Path, "/track/raceline")

    def process(self, center, width):
        c = path_to_xy(center)
        race, alpha = min_curvature_raceline(c, msg_to_array(width), self.margin)
        self.get_logger().info(f"raceline: curvature energy {curvature_energy(c):.3f} -> {curvature_energy(race):.3f}, "
                               f"max offset {np.abs(alpha).max():.1f} m")
        self.pub.publish(xy_to_path(race, self.get_clock().now().to_msg()))


def main():
    run(RacelineNode)
