"""Tiny road network with known answers, for engine tests.

    hub(0) --e0-- 1 --e1-- 2 --e2(bridge)-- 3 (village B)
                  |
                  e3
                  |
                  4 (village A)

e1 is low (hand_min 0.5 m): a 1 m water level cuts it, so node 2 and village B are isolated
by car, while village A stays reachable.
"""

import numpy as np

from evaku_ai.engine.graph import RoadGraph


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


HUB_NODES = np.array([0])
