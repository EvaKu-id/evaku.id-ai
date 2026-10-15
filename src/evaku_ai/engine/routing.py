import itertools

import numpy as np

from evaku_ai.core.errors import InvalidInputError
from evaku_ai.domain.arrays import BoolArray, FloatArray, IntArray
from evaku_ai.engine.accessibility import NO_NODE
from evaku_ai.engine.graph import RoadGraph


def path_from_predecessors(predecessors: IntArray, target: int) -> list[int]:
    path = [target]
    node = int(predecessors[target])
    while node != NO_NODE:
        if len(path) > predecessors.shape[0]:
            raise InvalidInputError("predecessor chain loops")
        path.append(node)
        node = int(predecessors[node])
    path.reverse()
    return path


def edge_lookup(
    graph: RoadGraph, passable: BoolArray, weights: FloatArray
) -> dict[tuple[int, int], int]:
    best: dict[tuple[int, int], int] = {}
    for edge in np.flatnonzero(passable).tolist():
        u, v = int(graph.edge_u[edge]), int(graph.edge_v[edge])
        key = (min(u, v), max(u, v))
        if key not in best or weights[edge] < weights[best[key]]:
            best[key] = edge
    return best


def path_edges(nodes: list[int], lookup: dict[tuple[int, int], int]) -> list[int]:
    edges = []
    for a, b in itertools.pairwise(nodes):
        key = (min(a, b), max(a, b))
        if key not in lookup:
            raise InvalidInputError(f"no usable edge between nodes {a} and {b}")
        edges.append(lookup[key])
    return edges


def boat_point(
    reachable: BoolArray, node_x: FloatArray, node_y: FloatArray, target: int
) -> tuple[int, float]:
    raise NotImplementedError
