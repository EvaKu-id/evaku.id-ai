"""Tiny road network with known answers, for engine tests.

    hub(0) --e0-- 1 --e1-- 2 --e2(bridge)-- 3
                  |
                  e3
                  |
                  4

Village A is node 4; village B is nodes 2 and 3 (100 people each).
e1 is low (hand_min 0.5 m): a 1 m water level cuts it, so node 2 and village B are isolated
by car, while village A stays reachable. small_geometry gives e3 a bend so route joins are tested.
"""

import numpy as np

from evaku_ai.engine.accessibility import NO_NODE, NodePopulation
from evaku_ai.engine.graph import RoadGeometry, RoadGraph
from evaku_ai.engine.simulation import ScenarioData


def small_graph() -> RoadGraph:
    return RoadGraph(
        edge_u=np.array([0, 1, 2, 1]),
        edge_v=np.array([1, 2, 3, 4]),
        travel_s=np.array([60.0, 60.0, 30.0, 90.0]),
        length_m=np.array([1000.0, 1000.0, 500.0, 1500.0]),
        hand_min=np.array([5.0, 0.5, 3.0, 4.0]),
        is_bridge=np.array([False, False, True, False]),
        node_x=np.array([0.0, 1000.0, 2000.0, 2500.0, 1000.0]),
        node_y=np.array([0.0, 0.0, 0.0, 0.0, -1500.0]),
    )


NODE_LON = np.array([98.0, 98.01, 98.02, 98.025, 98.01])
NODE_LAT = np.array([4.0, 4.0, 4.0, 4.0, 3.985])
E3_BEND = (98.012, 3.992)


def small_geometry() -> RoadGeometry:
    nodes = np.column_stack([NODE_LON, NODE_LAT])
    coords = [
        nodes[0],
        nodes[1],
        nodes[1],
        nodes[2],
        nodes[2],
        nodes[3],
        nodes[1],
        np.array(E3_BEND),
        nodes[4],
    ]
    return RoadGeometry(
        node_lon=NODE_LON.copy(),
        node_lat=NODE_LAT.copy(),
        edge_offsets=np.array([0, 2, 4, 6, 9]),
        edge_coords=np.array(coords),
    )


HUB_NODES = np.array([0])
VILLAGE_IDS = ["A", "B"]


def small_population() -> NodePopulation:
    return NodePopulation(
        village=np.array([NO_NODE, NO_NODE, 1, 1, 0]),
        total=np.array([0.0, 0.0, 100.0, 100.0, 50.0]),
        under5=np.array([0.0, 0.0, 10.0, 12.0, 5.0]),
        elderly=np.array([0.0, 0.0, 8.0, 9.0, 4.0]),
    )


def small_scenario(**changes: object) -> ScenarioData:
    fields = {
        "graph": small_graph(),
        "edge_ids": np.array([100, 101, 102, 103]),
        "rivers": ["r1", "r2"],
        "edge_river": np.zeros(4, dtype=int),
        "node_river": np.zeros(5, dtype=int),
        "node_hand": np.array([6.0, 5.0, 0.8, 3.0, 4.0]),
        "population": small_population(),
        "village_ids": VILLAGE_IDS,
        "hub_nodes": {10: 0},
        "health_nodes": np.array([4]),
        "health_hand": np.array([4.0]),
        "health_river": np.array([0]),
    }
    return ScenarioData(**(fields | changes))
