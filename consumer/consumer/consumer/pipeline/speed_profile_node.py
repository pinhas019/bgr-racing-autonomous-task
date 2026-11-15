"""Stage 3: path -> speed profile (point-mass model with a g-g ellipse), for both centerline and raceline.

1. Cornering limit:   v <= sqrt(a_lat / |kappa|), capped at v_top.
2. Forward pass:      accelerate with whatever grip is left over from cornering.
3. Backward pass:     same for braking. Closed loop -> each pass runs over two laps to converge.
"""
import numpy as np
from nav_msgs.msg import Path
from std_msgs.msg import Float64MultiArray

from consumer.pipeline.msg_utils import LATCHED, Stage, array_to_msg, path_to_xy, run
from consumer.pipeline.raceline_node import curvature, segment_lengths


def speed_profile(xy, a_lat, a_acc, a_brk, v_top):
    k = np.abs(curvature(xy))
    ds = segment_lengths(xy)
    v = np.minimum(v_top, np.sqrt(a_lat / np.maximum(k, 1e-9)))
    n = len(v)

    def reach(v0, k0, a_max, d):  # speed reachable after d metres, using the grip not spent on cornering
        lat_use = min(1.0, v0 * v0 * k0 / a_lat)
        return np.sqrt(v0 * v0 + 2 * a_max * np.sqrt(1 - lat_use ** 2) * d)

    for i in range(2 * n):  # forward: acceleration limit
        a, b = i % n, (i + 1) % n
        v[b] = min(v[b], reach(v[a], k[a], a_acc, ds[a]))
    for i in range(2 * n, 0, -1):  # backward: braking limit
        a, b = (i - 1) % n, i % n
        v[a] = min(v[a], reach(v[b], k[b], a_brk, ds[a]))
    return v


def lap_time(xy, v):
    return float(np.sum(segment_lengths(xy) / ((v + np.roll(v, -1)) / 2)))


class SpeedProfileNode(Stage):
    def __init__(self):
        super().__init__("speed_profile_node", {"center": (Path, "/track/centerline"),
                                                "race": (Path, "/track/raceline")})
        self.car = {k: self.declare_parameter(k, d).value
                    for k, d in {"a_lat": 15.0, "a_acc": 8.0, "a_brk": 12.0, "v_top": 30.0}.items()}
        self.pubs = {k: self.create_publisher(Float64MultiArray, f"/speed/{k}", LATCHED)
                     for k in ("centerline", "raceline")}

    def process(self, center, race):
        for key, msg in (("centerline", center), ("raceline", race)):
            xy = path_to_xy(msg)
            v = speed_profile(xy, **self.car)
            self.get_logger().info(f"{key}: lap {lap_time(xy, v):.2f} s, v {v.min():.1f}-{v.max():.1f} m/s")
            self.pubs[key].publish(array_to_msg(v))


def main():
    run(SpeedProfileNode)
