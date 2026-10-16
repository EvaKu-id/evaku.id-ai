import numpy as np

from evaku_ai.domain.arrays import BoolArray, FloatArray, IntArray
from evaku_ai.engine.accessibility import NodePopulation
from evaku_ai.engine.graph import RoadGeometry, RoadGraph, straight_geometry
from evaku_ai.engine.simulation import ScenarioData

GRID = 30
SPACING_M = 300.0
SPEED_M_PER_S = 30_000 / 3600
RIVER_COL = 14
BRIDGE_ROWS = (5, 15, 25)
BRIDGE_DECK_HAND_M = 5.0
HAND_BASE_M = 0.3
HAND_PER_M = 0.4 / SPACING_M
HAND_NOISE_M = 0.2
VILLAGE_BLOCK = 10
PEOPLE_PER_NODE_MIN = 20.0
PEOPLE_PER_NODE_MAX = 80.0
UNDER5_SHARE = 0.10
ELDERLY_SHARE = 0.08
HUBS = {1: (0, 0), 2: (GRID - 1, 0)}
HEALTH_CELLS = ((5, 5), (15, 25), (25, 20))
RIVER_ID = "synthetic-river"
SEED = 7
ORIGIN_LON = 98.0
ORIGIN_LAT = 4.3
M_PER_DEG = 111_320.0


def synthetic_scenario() -> ScenarioData:
    rng = np.random.default_rng(SEED)
    rows, cols = np.divmod(np.arange(GRID * GRID), GRID)
    node_x, node_y = cols * SPACING_M, rows * SPACING_M
    edge_u, edge_v, is_bridge = _edges()
    n_nodes, n_edges = GRID * GRID, edge_u.shape[0]

    mid_x = (node_x[edge_u] + node_x[edge_v]) / 2
    hand_min = _hand(mid_x, rng)
    hand_min[is_bridge] = BRIDGE_DECK_HAND_M
    length_m = np.full(n_edges, SPACING_M)

    total = rng.uniform(PEOPLE_PER_NODE_MIN, PEOPLE_PER_NODE_MAX, size=n_nodes)
    blocks_per_side = GRID // VILLAGE_BLOCK
    village = (rows // VILLAGE_BLOCK) * blocks_per_side + cols // VILLAGE_BLOCK
    health_nodes = np.array([_node(r, c) for r, c in HEALTH_CELLS])

    return ScenarioData(
        graph=RoadGraph(
            edge_u=edge_u,
            edge_v=edge_v,
            travel_s=length_m / SPEED_M_PER_S,
            length_m=length_m,
            hand_min=hand_min,
            is_bridge=is_bridge,
            node_x=node_x,
            node_y=node_y,
        ),
        edge_ids=np.arange(n_edges, dtype=np.int64),
        rivers=[RIVER_ID],
        edge_river=np.zeros(n_edges, dtype=np.int64),
        node_river=np.zeros(n_nodes, dtype=np.int64),
        node_hand=_hand(node_x, rng),
        population=NodePopulation(
            village=village.astype(np.int64),
            total=total,
            under5=total * UNDER5_SHARE,
            elderly=total * ELDERLY_SHARE,
        ),
        village_ids=[f"SYN-{i + 1:02d}" for i in range(blocks_per_side**2)],
        hub_nodes={hub_id: _node(r, c) for hub_id, (r, c) in HUBS.items()},
        health_nodes=health_nodes,
        health_hand=_hand(node_x[health_nodes], rng),
        health_river=np.zeros(health_nodes.shape[0], dtype=np.int64),
    )


def synthetic_geometry(data: ScenarioData) -> RoadGeometry:
    graph = data.graph
    return straight_geometry(
        graph, ORIGIN_LON + graph.node_x / M_PER_DEG, ORIGIN_LAT + graph.node_y / M_PER_DEG
    )


def _edges() -> tuple[IntArray, IntArray, BoolArray]:
    u, v, bridge = [], [], []
    for r in range(GRID):
        for c in range(GRID):
            if c + 1 < GRID and (c != RIVER_COL or r in BRIDGE_ROWS):
                u.append(_node(r, c))
                v.append(_node(r, c + 1))
                bridge.append(c == RIVER_COL)
            if r + 1 < GRID:
                u.append(_node(r, c))
                v.append(_node(r + 1, c))
                bridge.append(False)
    return np.array(u, dtype=np.int64), np.array(v, dtype=np.int64), np.array(bridge)


def _hand(x: FloatArray, rng: np.random.Generator) -> FloatArray:
    river_x = (RIVER_COL + 0.5) * SPACING_M
    noise = rng.uniform(-HAND_NOISE_M, HAND_NOISE_M, x.shape[0])
    return np.maximum(0.0, HAND_BASE_M + np.abs(x - river_x) * HAND_PER_M + noise)


def _node(row: int, col: int) -> int:
    return row * GRID + col
