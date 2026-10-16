from dataclasses import dataclass

import numpy as np
from scipy import sparse

from evaku_ai.core.errors import InvalidInputError
from evaku_ai.domain.arrays import BoolArray, FloatArray, IntArray


@dataclass(frozen=True)
class RoadGraph:
    # node_x/node_y are projected UTM meters; hand_min is NaN outside the DEM.
    edge_u: IntArray
    edge_v: IntArray
    travel_s: FloatArray
    length_m: FloatArray
    hand_min: FloatArray
    is_bridge: BoolArray
    node_x: FloatArray
    node_y: FloatArray

    def __post_init__(self) -> None:
        edge_arrays = (
            self.edge_u,
            self.edge_v,
            self.travel_s,
            self.length_m,
            self.hand_min,
            self.is_bridge,
        )
        if any(a.ndim != 1 or a.shape != self.edge_u.shape for a in edge_arrays):
            raise InvalidInputError("edge arrays must be 1-D with the same length")
        if self.node_x.ndim != 1 or self.node_x.shape != self.node_y.shape:
            raise InvalidInputError("node_x and node_y must be 1-D with the same length")
        same_int_dtype = self.edge_v.dtype == self.edge_u.dtype
        if not (np.issubdtype(self.edge_u.dtype, np.integer) and same_int_dtype):
            raise InvalidInputError("edge_u and edge_v must share an integer dtype")
        if self.is_bridge.dtype != np.bool_:
            raise InvalidInputError("is_bridge must be a bool array")
        if self.n_edges:
            low = min(self.edge_u.min(), self.edge_v.min())
            high = max(self.edge_u.max(), self.edge_v.max())
            if low < 0 or high >= self.n_nodes:
                raise InvalidInputError(f"node index out of range 0..{self.n_nodes - 1}")
        # Zero weights would vanish from the sparse matrix and silently disconnect the edge.
        if not np.all(np.isfinite(self.travel_s) & (self.travel_s > 0)):
            raise InvalidInputError("travel_s must be finite and > 0")

    @property
    def n_nodes(self) -> int:
        return int(self.node_x.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_u.shape[0])


ENDPOINT_TOLERANCE_DEG = 1e-6
MAX_LON = 180.0
MAX_LAT = 90.0


@dataclass(frozen=True)
class RoadGeometry:
    # Lon/lat in EPSG:4326. Edge i runs from node edge_u[i] to edge_v[i] along
    # edge_coords[edge_offsets[i]:edge_offsets[i + 1]].
    node_lon: FloatArray
    node_lat: FloatArray
    edge_offsets: IntArray
    edge_coords: FloatArray

    def __post_init__(self) -> None:
        if self.node_lon.ndim != 1 or self.node_lon.shape != self.node_lat.shape:
            raise InvalidInputError("node_lon and node_lat must be 1-D with the same length")
        offsets = self.edge_offsets
        if offsets.ndim != 1 or not offsets.size or not np.issubdtype(offsets.dtype, np.integer):
            raise InvalidInputError("edge_offsets must be a non-empty 1-D integer array")
        if self.edge_coords.ndim != 2 or self.edge_coords.shape[1] != 2:
            raise InvalidInputError("edge_coords must have shape (n, 2)")
        if offsets[0] != 0 or offsets[-1] != self.edge_coords.shape[0]:
            raise InvalidInputError("edge_offsets must start at 0 and end at the number of coords")
        if np.any(np.diff(offsets) < 2):
            raise InvalidInputError("every edge needs at least two coordinates")
        lon = np.concatenate([self.node_lon, self.edge_coords[:, 0]])
        lat = np.concatenate([self.node_lat, self.edge_coords[:, 1]])
        finite = np.all(np.isfinite(lon)) and np.all(np.isfinite(lat))
        if not finite or np.any(np.abs(lon) > MAX_LON) or np.any(np.abs(lat) > MAX_LAT):
            raise InvalidInputError("coordinates must be finite lon/lat in EPSG:4326")

    @property
    def n_nodes(self) -> int:
        return int(self.node_lon.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_offsets.shape[0] - 1)


def check_geometry(graph: RoadGraph, geometry: RoadGeometry) -> None:
    if geometry.n_nodes != graph.n_nodes or geometry.n_edges != graph.n_edges:
        raise InvalidInputError(
            f"geometry has {geometry.n_nodes} nodes and {geometry.n_edges} edges, "
            f"graph has {graph.n_nodes} and {graph.n_edges}"
        )
    nodes = np.column_stack([geometry.node_lon, geometry.node_lat])
    first = geometry.edge_coords[geometry.edge_offsets[:-1]]
    last = geometry.edge_coords[geometry.edge_offsets[1:] - 1]
    starts = np.allclose(first, nodes[graph.edge_u], rtol=0, atol=ENDPOINT_TOLERANCE_DEG)
    ends = np.allclose(last, nodes[graph.edge_v], rtol=0, atol=ENDPOINT_TOLERANCE_DEG)
    if not (starts and ends):
        raise InvalidInputError("edge geometry must run from node edge_u to node edge_v")


def straight_geometry(graph: RoadGraph, node_lon: FloatArray, node_lat: FloatArray) -> RoadGeometry:
    nodes = np.column_stack([node_lon, node_lat]).astype(np.float64)
    ends = np.stack([nodes[graph.edge_u], nodes[graph.edge_v]], axis=1)
    return RoadGeometry(
        node_lon=np.asarray(node_lon, dtype=np.float64),
        node_lat=np.asarray(node_lat, dtype=np.float64),
        edge_offsets=np.arange(0, 2 * graph.n_edges + 1, 2, dtype=np.int64),
        edge_coords=ends.reshape(-1, 2),
    )


def to_csr(
    graph: RoadGraph, passable: BoolArray | None = None, weights: FloatArray | None = None
) -> sparse.csr_matrix:
    keep = _edge_mask(passable, graph.n_edges)
    cost = graph.travel_s if weights is None else _check_weights(weights, graph.n_edges)
    u, v, w = graph.edge_u[keep], graph.edge_v[keep], cost[keep]
    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])
    data = np.concatenate([w, w])

    # Building a CSR from COO sums duplicate entries, so keep only the cheapest parallel edge.
    order = np.lexsort((data, cols, rows))
    rows, cols, data = rows[order], cols[order], data[order]
    first = np.ones(rows.shape[0], dtype=bool)
    first[1:] = (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])

    shape = (graph.n_nodes, graph.n_nodes)
    return sparse.csr_matrix((data[first], (rows[first], cols[first])), shape=shape)


def _check_weights(weights: FloatArray, n_edges: int) -> FloatArray:
    cost = np.asarray(weights, dtype=np.float64)
    if cost.shape != (n_edges,):
        raise InvalidInputError(f"weights must have length {n_edges}")
    # Zero weights would vanish from the sparse matrix and silently disconnect the edge.
    if not np.all(np.isfinite(cost) & (cost > 0)):
        raise InvalidInputError("weights must be finite and > 0")
    return cost


def _edge_mask(passable: BoolArray | None, n_edges: int) -> BoolArray:
    if passable is None:
        return np.ones(n_edges, dtype=bool)
    if passable.dtype != np.bool_ or passable.shape != (n_edges,):
        raise InvalidInputError(f"passable must be a bool array of length {n_edges}")
    return passable
