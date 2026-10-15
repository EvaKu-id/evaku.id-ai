import numpy as np
import pytest

from evaku_ai.core.errors import InvalidInputError
from evaku_ai.engine.accessibility import NO_NODE
from evaku_ai.engine.graph import RoadGraph
from evaku_ai.engine.routing import edge_lookup, path_edges, path_from_predecessors


def test_path_from_predecessors():
    predecessors = np.array([NO_NODE, 0, 1, NO_NODE])
    assert path_from_predecessors(predecessors, 2) == [0, 1, 2]
    assert path_from_predecessors(predecessors, 0) == [0]


def test_path_from_predecessors_rejects_loops():
    with pytest.raises(InvalidInputError, match="loops"):
        path_from_predecessors(np.array([1, 0]), 0)


def _graph():
    return RoadGraph(
        edge_u=np.array([0, 1, 1]),
        edge_v=np.array([1, 0, 2]),
        travel_s=np.ones(3),
        length_m=np.array([500.0, 300.0, 100.0]),
        hand_min=np.zeros(3),
        is_bridge=np.zeros(3, dtype=bool),
        node_x=np.zeros(3),
        node_y=np.zeros(3),
    )


def test_path_edges_uses_the_shortest_usable_edge():
    graph = _graph()
    lookup = edge_lookup(graph, np.ones(3, dtype=bool), graph.length_m)
    assert path_edges([0, 1, 2], lookup) == [1, 2]
    lookup = edge_lookup(graph, np.array([True, False, True]), graph.length_m)
    assert path_edges([2, 1, 0], lookup) == [2, 0]
    assert path_edges([1], lookup) == []


def test_path_edges_rejects_missing_edges():
    graph = _graph()
    lookup = edge_lookup(graph, np.array([True, True, False]), graph.length_m)
    with pytest.raises(InvalidInputError, match="no usable edge"):
        path_edges([0, 1, 2], lookup)


def test_edge_lookup_keeps_the_shorter_edge_seen_first():
    graph = RoadGraph(
        edge_u=np.array([0, 0]),
        edge_v=np.array([1, 1]),
        travel_s=np.ones(2),
        length_m=np.array([100.0, 900.0]),
        hand_min=np.zeros(2),
        is_bridge=np.zeros(2, dtype=bool),
        node_x=np.zeros(2),
        node_y=np.zeros(2),
    )
    assert edge_lookup(graph, np.ones(2, dtype=bool), graph.length_m) == {(0, 1): 0}
