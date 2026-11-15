"""Stage 3: path -> speed profile (point-mass model), for both centerline and raceline.

With constant accel/brake limits the forward/backward passes have a closed form (no Python loops):
    forward:  v_i^2 = min_{k<=i} (vlim_k^2 + 2 a_acc (s_i - s_k))  = 2 a_acc s_i + cummin(vlim^2 - 2 a_acc s)
    backward: v_i^2 = min_{k>=i} (vlim_k^2 + 2 a_brk (s_k - s_i))  = reverse cummin of (vlim^2 + 2 a_brk s) - 2 a_brk s
and min(forward, backward) equals the usual sequential forward-then-backward result.
The lap is closed, so the profile is computed over three unrolled laps and the middle one is kept.
"""
import numpy as np
from nav_msgs.msg import Path
from std_msgs.msg import Float64MultiArray

from consumer.pipeline.msg_utils import Stage, array_to_msg, path_to_xy, require_params, run
from consumer.pipeline.raceline_node import curvature, segment_lengths

CAR_PARAMS = ("a_lat", "a_acc", "a_brk", "v_top")


def speed_profile(xy, a_lat, a_acc, a_brk, v_top):
    n = len(xy)
    vlim2 = np.tile(np.minimum(v_top ** 2, a_lat / np.maximum(np.abs(curvature(xy)), 1e-12)), 3)
    s = np.concatenate([[0.0], np.cumsum(np.tile(segment_lengths(xy), 3))[:-1]])
    fwd = 2 * a_acc * s + np.minimum.accumulate(vlim2 - 2 * a_acc * s)
    bwd = np.minimum.accumulate((vlim2 + 2 * a_brk * s)[::-1])[::-1] - 2 * a_brk * s
    return np.sqrt(np.minimum(fwd, bwd)[n:2 * n])


def lap_time(xy, v):
    return float(np.sum(segment_lengths(xy) / ((v + np.roll(v, -1)) / 2)))


class SpeedProfileNode(Stage):
    def __init__(self):
        super().__init__("speed_profile_node", {"center": (Path, "/track/centerline"),
                                                "race": (Path, "/track/raceline")})
        self.car = require_params(self, *CAR_PARAMS)
        self.pubs = {k: self.output(Float64MultiArray, f"/speed/{k}") for k in ("centerline", "raceline")}

    def process(self, center, race):
        for key, msg in (("centerline", center), ("raceline", race)):
            xy = path_to_xy(msg)
            v = speed_profile(xy, **self.car)
            self.get_logger().info(f"{key}: lap {lap_time(xy, v):.2f} s, v {v.min():.1f}-{v.max():.1f} m/s")
            self.pubs[key].publish(array_to_msg(v))


def main():
    run(SpeedProfileNode)
