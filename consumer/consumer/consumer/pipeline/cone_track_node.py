"""Bridge from perception to planning: LiDAR cone detections -> left / right track boundaries.

The LiDAR sees no colour, so which cone is on which side has to come from geometry alone:
1. Track cones = cone candidates whose top height matches the track's cone type (config).
2. Delaunay triangulation of the cones (scipy.spatial.Delaunay). An edge can be a *gate* (one cone on each
   side of the track) if it is about one track width long.
3. Starting at the car, walk forward from gate to gate: the next gate is the nearest gate midpoint ahead that
   is reachable (max step, max turn) and whose edge lies ACROSS the direction of travel. Edges between two
   cones of the same boundary run ALONG it, so they are rejected even when they have track-width length.
4. Each gate's cones are put on the left or right of the travel direction; in walk order they form the
   left and right boundaries, published as nav_msgs/Path exactly like the map-based track.
"""
import numpy as np
from nav_msgs.msg import Path
from scipy.spatial import Delaunay
from std_msgs.msg import Float64MultiArray

from consumer.pipeline.lidar_cluster_node import COLUMNS
from consumer.pipeline.msg_utils import Stage, msg_to_array, require_params, run, xy_to_path


def unit(v):
    return v / np.linalg.norm(v)


def gate_edges(cones, min_width, max_width):
    """Delaunay edges (i, j) whose length could be a track width."""
    if len(cones) < 3:
        return np.empty((0, 2), int)
    tri = Delaunay(cones)
    edges = np.sort(np.vstack([tri.simplices[:, [0, 1]], tri.simplices[:, [1, 2]], tri.simplices[:, [0, 2]]]), axis=1)
    edges = np.unique(edges, axis=0)
    length = np.linalg.norm(cones[edges[:, 0]] - cones[edges[:, 1]], axis=1)
    return edges[(length >= min_width) & (length <= max_width)]


def walk_gates(cones, edges, start, heading, max_step, max_turn_deg, max_gate_skew_deg):
    """Ordered list of gates (i, j) from the car forward."""
    mids = cones[edges].mean(axis=1)
    pos, heading, used, route = np.asarray(start, float), unit(np.asarray(heading, float)), set(), []
    while True:
        best, best_dist = None, np.inf
        for k, (mid, (i, j)) in enumerate(zip(mids, edges)):
            step = mid - pos
            dist = np.linalg.norm(step)
            if k in used or dist < 1e-6 or dist > max_step:
                continue
            turn = np.degrees(np.arccos(np.clip(np.dot(unit(step), heading), -1, 1)))
            across = abs(np.dot(unit(cones[j] - cones[i]), unit(step)))  # 0 = perpendicular to travel
            if turn <= max_turn_deg and across <= np.sin(np.radians(max_gate_skew_deg)) and dist < best_dist:
                best, best_dist = k, dist
        if best is None:
            return route
        used.add(best)
        heading, pos = unit(mids[best] - pos), mids[best]
        route.append((tuple(edges[best]), heading))


def split_sides(cones, route):
    """Left / right cone indices in driving order (each cone once, at its first gate)."""
    left, right = [], []
    for (i, j), heading in route:
        mid = (cones[i] + cones[j]) / 2
        for c in (i, j):
            side = left if np.cross(heading, cones[c] - mid) > 0 else right
            if c not in left and c not in right:
                side.append(c)
    return left, right


def cone_boundaries(objects, track_cone_min_height, track_cone_max_height, gate_min_width, gate_max_width,
                    max_step, max_turn_deg, max_gate_skew_deg):
    """Return (left boundary xy, right boundary xy, all track cones xy, gates)."""
    col = COLUMNS.index
    top = objects[:, col("top_height")]
    keep = (objects[:, col("is_cone")] == 1) & (top >= track_cone_min_height) & (top <= track_cone_max_height)
    cones = objects[keep][:, [col("x"), col("y")]]
    edges = gate_edges(cones, gate_min_width, gate_max_width)
    route = walk_gates(cones, edges, start=(0.0, 0.0), heading=(1.0, 0.0), max_step=max_step,
                       max_turn_deg=max_turn_deg, max_gate_skew_deg=max_gate_skew_deg)
    left, right = split_sides(cones, route)
    return cones[left], cones[right], cones, [g for g, _ in route]


class ConeTrackNode(Stage):
    def __init__(self):
        super().__init__("cone_track_node", {"objects": (Float64MultiArray, "/lidar/clusters")})
        self.cfg = require_params(self, "track_cone_min_height", "track_cone_max_height", "gate_min_width",
                                  "gate_max_width", "max_step", "max_turn_deg", "max_gate_skew_deg", "min_gates")
        self.min_gates = self.cfg.pop("min_gates")
        self.pub_left = self.output(Path, "/cones/left_boundary")
        self.pub_right = self.output(Path, "/cones/right_boundary")

    def process(self, objects):
        left, right, cones, gates = cone_boundaries(msg_to_array(objects), **self.cfg)
        self.get_logger().info(f"{len(cones)} track cones, {len(gates)} gates: left {len(left)} cones, "
                               f"right {len(right)} cones")
        if len(gates) < self.min_gates or len(left) < 2 or len(right) < 2:
            raise RuntimeError(f"no cone track found ({len(gates)} gates; need {self.min_gates} and 2 cones a side)")
        stamp = self.get_clock().now().to_msg()
        self.pub_left.publish(xy_to_path(left, stamp, frame="lidar"))
        self.pub_right.publish(xy_to_path(right, stamp, frame="lidar"))


def main():
    run(ConeTrackNode)
