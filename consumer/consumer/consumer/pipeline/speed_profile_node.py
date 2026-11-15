"""Planning stage 3: path -> speed profile (point-mass model), for both centerline and raceline.

With constant accel/brake limits the forward/backward passes have a closed form (no Python loops):
    forward:  v_i^2 = min_{k<=i} (vlim_k^2 + 2 a_acc (s_i - s_k))  = 2 a_acc s_i + cummin(vlim^2 - 2 a_acc s)
    backward: v_i^2 = min_{k>=i} (vlim_k^2 + 2 a_brk (s_k - s_i))  = reverse cummin of (vlim^2 + 2 a_brk s) - 2 a_brk s
and min(forward, backward) equals the usual sequential forward-then-backward result.
A closed lap is unrolled three times and the middle lap kept (so the start/finish wraps correctly).
An open path starts at `v_start` (the speed the car has where the path begins) and ends unconstrained.
"""
import numpy as np
from nav_msgs.msg import Path
from std_msgs.msg import Float64MultiArray

from consumer.pipeline.msg_utils import Stage, array_to_msg, path_to_xy, require_params, run
from consumer.pipeline.raceline_node import curvature, segment_lengths

CAR_PARAMS = ("a_lat", "a_acc", "a_brk", "v_top")


def speed_profile(xy, a_lat, a_acc, a_brk, v_top, closed=True, v_start=None):
    n, laps = len(xy), 3 if closed else 1
    vlim2 = np.tile(np.minimum(v_top ** 2, a_lat / np.maximum(np.abs(curvature(xy, closed)), 1e-12)), laps)
    if not closed:
        vlim2[0] = min(vlim2[0], v_start ** 2)
    s = np.concatenate([[0.0], np.cumsum(np.tile(segment_lengths(xy, closed), laps))[:-1]])
    fwd = 2 * a_acc * s + np.minimum.accumulate(vlim2 - 2 * a_acc * s)
    bwd = np.minimum.accumulate((vlim2 + 2 * a_brk * s)[::-1])[::-1] - 2 * a_brk * s
    v = np.sqrt(np.maximum(np.minimum(fwd, bwd), 0.0))
    return v[n:2 * n] if closed else v


def lap_time(xy, v, closed=True):
    """Time to drive the path: a full lap if closed, start to end if open (the last ds is 0 there)."""
    mean_v = np.maximum((v + np.roll(v, -1)) / 2, 1e-9)
    return float(np.sum(segment_lengths(xy, closed) / mean_v))


class SpeedProfileNode(Stage):
    def __init__(self):
        super().__init__("speed_profile_node", {"center": (Path, "centerline"), "race": (Path, "raceline")})
        self.car = require_params(self, *CAR_PARAMS, "closed")
        if not self.car["closed"]:
            self.car.update(require_params(self, "v_start"))
        self.pubs = {k: self.output(Float64MultiArray, f"speed/{k}") for k in ("centerline", "raceline")}

    def process(self, center, race):
        for key, msg in (("centerline", center), ("raceline", race)):
            xy = path_to_xy(msg)
            v = speed_profile(xy, **self.car)
            t = lap_time(xy, v, self.car["closed"])
            self.get_logger().info(f"{key}: time {t:.2f} s, v {v.min():.1f}-{v.max():.1f} m/s")
            self.pubs[key].publish(array_to_msg(v))


def main():
    run(SpeedProfileNode)
