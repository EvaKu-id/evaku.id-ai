from dataclasses import replace

import numpy as np
import pytest

from evaku_ai.core.errors import InvalidInputError
from evaku_ai.engine.graph import RoadGraph, to_csr
from tests.fixtures.small_graph import small_graph


def test_csr_is_symmetric_with_travel_times():
    csr = to_csr(small_graph())
    assert csr.shape == (5, 5)
    assert csr.nnz == 8
    assert csr[0, 1] == csr[1, 0] == 60.0
    assert csr[2, 3] == csr[3, 2] == 30.0
    assert csr[1, 4] == csr[4, 1] == 90.0


def test_mask_removes_edge_in_both_directions():
    csr = to_csr(small_graph(), passable=np.array([True, False, True, True]))
    assert csr.nnz == 6
    assert csr[1, 2] == csr[2, 1] == 0


def test_parallel_edges_keep_fastest():
    graph = RoadGraph(
        edge_u=np.array([0, 1, 0]),
        edge_v=np.array([1, 0, 1]),
        travel_s=np.array([60.0, 40.0, 50.0]),
        length_m=np.array([1000.0, 900.0, 950.0]),
        hand_min=np.array([5.0, 5.0, 5.0]),
        is_bridge=np.array([False, False, False]),
        node_x=np.array([0.0, 1000.0]),
        node_y=np.array([0.0, 0.0]),
    )
    csr = to_csr(graph)
    assert csr.nnz == 2
    assert csr[0, 1] == csr[1, 0] == 40.0


def test_masked_parallel_edge_falls_back_to_the_other():
    graph = replace(
        small_graph(),
        edge_u=np.array([0, 1, 2, 0]),
        edge_v=np.array([1, 2, 3, 1]),
        travel_s=np.array([60.0, 60.0, 30.0, 10.0]),
    )
    csr = to_csr(graph, passable=np.array([True, True, True, False]))
    assert csr[0, 1] == 60.0


def test_empty_graph():
    graph = RoadGraph(
        edge_u=np.array([], dtype=int),
        edge_v=np.array([], dtype=int),
        travel_s=np.array([]),
        length_m=np.array([]),
        hand_min=np.array([]),
        is_bridge=np.array([], dtype=bool),
        node_x=np.array([0.0]),
        node_y=np.array([0.0]),
    )
    assert to_csr(graph).nnz == 0


@pytest.mark.parametrize(
    "mask",
    [np.array([True, True, True]), np.array([1, 1, 1, 1]), np.ones((2, 2), dtype=bool)],
)
def test_mask_must_be_bool_per_edge(mask):
    with pytest.raises(InvalidInputError, match="passable"):
        to_csr(small_graph(), passable=mask)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"travel_s": np.array([60.0, 60.0, 30.0])}, "same length"),
        ({"node_y": np.array([0.0, 0.0])}, "node_x and node_y"),
        ({"edge_u": np.array([0.0, 1.0, 2.0, 1.0])}, "integer"),
        ({"is_bridge": np.array([0, 0, 1, 0])}, "bool"),
        ({"edge_v": np.array([1, 2, 3, 5])}, "out of range"),
        ({"edge_u": np.array([-1, 1, 2, 1])}, "out of range"),
        ({"travel_s": np.array([60.0, 0.0, 30.0, 90.0])}, "travel_s"),
        ({"travel_s": np.array([60.0, np.nan, 30.0, 90.0])}, "travel_s"),
    ],
)
def test_graph_rejects_invalid_arrays(changes, message):
    with pytest.raises(InvalidInputError, match=message):
        replace(small_graph(), **changes)
