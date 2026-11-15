"""LiDAR stage 2: group obstacle points into objects and flag the cone-sized ones.

Euclidean clustering: points closer than `radius` are linked (scipy cKDTree.query_pairs) and every connected
component of that graph is one object (scipy.sparse.csgraph.connected_components). The sensor's ~0.1 deg
resolution keeps neighbouring points < 0.45 m apart even at 250 m, so one radius works at every range.
Per-object statistics use scipy.ndimage; a cluster is a cone candidate if its footprint and height match a cone.
"""
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64MultiArray, Int32MultiArray

from consumer.pipeline.lidar_ground_node import LABELS, height_above
from consumer.pipeline.msg_utils import (Stage, array_to_msg, cloud_to_array, msg_to_array, require_params,
                                         run)

COLUMNS = ["id", "n_points", "x", "y", "z", "size_x", "size_y", "size_z", "top_height", "range",
           "mean_reflectivity", "is_cone"]


def euclidean_clusters(xyz, radius, min_points):
    """Cluster id per point (0..k-1), or -1 for points in clusters smaller than min_points."""
    pairs = cKDTree(xyz).query_pairs(radius, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(xyz), len(xyz)))
    _, ids = connected_components(graph, directed=False)
    sizes = np.bincount(ids)
    keep = sizes >= min_points
    new_id = np.where(keep, np.cumsum(keep) - 1, -1)  # renumber surviving clusters 0..k-1
    return new_id[ids]


def describe_clusters(xyz, height, reflectivity, ids):
    """One row per cluster (columns: COLUMNS without is_cone)."""
    k = ids.max() + 1 if len(ids) and ids.max() >= 0 else 0
    index = np.arange(k)
    stat = lambda f, v: np.asarray(f(v, labels=ids, index=index), dtype=float).reshape(k)  # noqa: E731
    lo = np.column_stack([stat(ndimage.minimum, xyz[:, i]) for i in range(3)]).reshape(k, 3)
    hi = np.column_stack([stat(ndimage.maximum, xyz[:, i]) for i in range(3)]).reshape(k, 3)
    center = np.column_stack([stat(ndimage.mean, xyz[:, i]) for i in range(3)]).reshape(k, 3)
    return np.column_stack([
        index, np.bincount(ids[ids >= 0], minlength=k), center, hi - lo,
        stat(ndimage.maximum, height), np.hypot(center[:, 0], center[:, 1]), stat(ndimage.mean, reflectivity),
    ])


def cone_candidates(table, cone_max_footprint, cone_min_height, cone_max_height):
    footprint = np.maximum(table[:, COLUMNS.index("size_x")], table[:, COLUMNS.index("size_y")])
    top = table[:, COLUMNS.index("top_height")]
    return (footprint <= cone_max_footprint) & (top >= cone_min_height) & (top <= cone_max_height)


def cluster_obstacles(points, labels, plane, radius, min_points, cone_max_footprint, cone_min_height,
                      cone_max_height):
    """Return (per-pixel cluster id, -1 = none; cluster table with COLUMNS)."""
    obstacle = np.flatnonzero(labels == LABELS["obstacle"])
    xyz = np.column_stack([points["x"], points["y"], points["z"]]).astype(float)[obstacle]
    ids = euclidean_clusters(xyz, radius, min_points)
    table = describe_clusters(xyz, height_above(xyz, plane), points["reflectivity"][obstacle].astype(float), ids)
    table = np.column_stack([table, cone_candidates(table, cone_max_footprint, cone_min_height, cone_max_height)])
    pixel_ids = np.full(len(points), -1, dtype=np.int32)
    pixel_ids[obstacle] = ids
    return pixel_ids, table


class LidarClusterNode(Stage):
    def __init__(self):
        super().__init__("lidar_cluster_node", {"cloud": (PointCloud2, "/lidar/points"),
                                                "labels": (Int32MultiArray, "/lidar/labels"),
                                                "plane": (Float64MultiArray, "/lidar/ground_plane")})
        self.cfg = require_params(self, "radius", "min_points", "cone_max_footprint", "cone_min_height",
                                  "cone_max_height")
        self.pub_ids = self.output(Int32MultiArray, "/lidar/cluster_ids")
        self.pub_table = self.output(Float64MultiArray, "/lidar/clusters")

    def process(self, cloud, labels, plane):
        pixel_ids, table = cluster_obstacles(cloud_to_array(cloud), msg_to_array(labels).ravel(),
                                             msg_to_array(plane), **self.cfg)
        n_cones = int(table[:, COLUMNS.index("is_cone")].sum())
        self.get_logger().info(f"{len(table)} objects, {n_cones} cone candidates")
        self.pub_ids.publish(array_to_msg(pixel_ids.reshape(cloud.height, cloud.width)))
        self.pub_table.publish(array_to_msg(table, columns=COLUMNS))


def main():
    run(LidarClusterNode)
