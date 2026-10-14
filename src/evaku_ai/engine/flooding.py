import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from evaku_ai.core.errors import InvalidInputError
from evaku_ai.domain.arrays import BoolArray, FloatArray, StatusArray
from evaku_ai.domain.enums import RoadCondition
from evaku_ai.domain.rules import PassabilityThresholds
from evaku_ai.engine.graph import RoadGraph

CM_PER_M = 100.0


def depth_from_water_level(
    hand_min: FloatArray, water_level_m: float, connected: BoolArray | None = None
) -> FloatArray:
    if not np.isfinite(water_level_m) or water_level_m < 0:
        raise InvalidInputError(f"water_level_m must be finite and >= 0, got {water_level_m}")
    depth = np.maximum(0.0, water_level_m - np.asarray(hand_min, dtype=float)) * CM_PER_M
    if connected is not None:
        if connected.dtype != np.bool_ or connected.shape != depth.shape:
            raise InvalidInputError("connected must be a bool array shaped like hand_min")
        depth[~connected] = 0.0
    return depth


def bridge_approach_depth(graph: RoadGraph, depth_cm: FloatArray) -> FloatArray:
    depth = np.asarray(depth_cm, dtype=float)
    if depth.shape != (graph.n_edges,):
        raise InvalidInputError(f"depth_cm must have length {graph.n_edges}")
    result = depth.copy()
    bridge = graph.is_bridge
    if not bridge.any():
        return result

    bridge_u, bridge_v = graph.edge_u[bridge], graph.edge_v[bridge]
    links = sparse.coo_matrix(
        (np.ones(bridge_u.shape[0]), (bridge_u, bridge_v)), shape=(graph.n_nodes, graph.n_nodes)
    )
    n_bridges, label = csgraph.connected_components(links, directed=False)

    on_bridge = np.zeros(graph.n_nodes, dtype=bool)
    on_bridge[bridge_u] = True
    on_bridge[bridge_v] = True
    road = ~bridge
    approach = np.full(n_bridges, np.nan)
    for ends in (graph.edge_u[road], graph.edge_v[road]):
        touches = on_bridge[ends]
        # fmax ignores NaN, so one unknown approach does not hide a known one.
        np.fmax.at(approach, label[ends[touches]], depth[road][touches])

    result[bridge] = approach[label[bridge_u]]
    return result


def road_status(
    depth_cm: FloatArray, thresholds: PassabilityThresholds = PassabilityThresholds()
) -> StatusArray:
    depth = np.asarray(depth_cm, dtype=float)
    status = np.full(depth.shape, RoadCondition.CUT, dtype=np.int8)
    status[depth <= thresholds.truck_cm] = RoadCondition.LIMITED
    status[depth <= thresholds.car_cm] = RoadCondition.SAFE
    status[np.isnan(depth)] = RoadCondition.UNKNOWN
    return status
