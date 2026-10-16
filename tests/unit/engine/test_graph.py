from dataclasses import replace

import numpy as np
import pytest

from evaku_ai.core.errors import InvalidInputError
from evaku_ai.engine.graph import (
    RoadGeometry,
    RoadGraph,
    check_geometry,
    straight_geometry,
    to_csr,
)
from tests.fixtures.small_graph import NODE_LAT, NODE_LON, small_geometry, small_graph


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


def test_custom_weights_pick_the_shortest_parallel_edge():
    graph = RoadGraph(
        edge_u=np.array([0, 1]),
        edge_v=np.array([1, 0]),
        travel_s=np.array([10.0, 50.0]),
        length_m=np.array([900.0, 400.0]),
        hand_min=np.zeros(2),
        is_bridge=np.zeros(2, dtype=bool),
        node_x=np.zeros(2),
        node_y=np.zeros(2),
    )
    assert to_csr(graph)[0, 1] == 10.0
    assert to_csr(graph, weights=graph.length_m)[0, 1] == 400.0


@pytest.mark.parametrize("weights", [np.array([1.0]), np.array([1.0, 0.0, 1.0, 1.0])])
def test_custom_weights_must_match_and_be_positive(weights):
    with pytest.raises(InvalidInputError, match="weights"):
        to_csr(small_graph(), weights=weights)


def _geometry(**changes):
    geometry = small_geometry()
    fields = {
        "node_lon": geometry.node_lon,
        "node_lat": geometry.node_lat,
        "edge_offsets": geometry.edge_offsets,
        "edge_coords": geometry.edge_coords,
    }
    return RoadGeometry(**(fields | changes))


def test_geometry_counts_nodes_and_edges():
    geometry = small_geometry()
    assert (geometry.n_nodes, geometry.n_edges) == (5, 4)
    check_geometry(small_graph(), geometry)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"node_lat": NODE_LAT[:4]}, "node_lon and node_lat"),
        ({"edge_offsets": np.array([], dtype=np.int64)}, "non-empty 1-D integer"),
        ({"edge_offsets": np.array([0.0, 2.0, 4.0, 6.0, 9.0])}, "non-empty 1-D integer"),
        ({"edge_coords": np.zeros((9, 3))}, "shape \\(n, 2\\)"),
        ({"edge_offsets": np.array([1, 2, 4, 6, 9])}, "start at 0"),
        ({"edge_offsets": np.array([0, 2, 4, 6, 8])}, "start at 0"),
        ({"edge_offsets": np.array([0, 1, 4, 6, 9])}, "at least two"),
        ({"node_lon": np.array([np.nan, 98.0, 98.0, 98.0, 98.0])}, "finite lon/lat"),
        ({"node_lat": np.array([95.0, 4.0, 4.0, 4.0, 4.0])}, "finite lon/lat"),
    ],
)
def test_geometry_rejects_invalid_arrays(changes, message):
    with pytest.raises(InvalidInputError, match=message):
        _geometry(**changes)


def test_check_geometry_rejects_mismatches():
    graph = small_graph()
    with pytest.raises(InvalidInputError, match="geometry has 4 nodes"):
        check_geometry(graph, _geometry(node_lon=NODE_LON[:4], node_lat=NODE_LAT[:4]))
    reversed_e0 = small_geometry().edge_coords.copy()
    reversed_e0[[0, 1]] = reversed_e0[[1, 0]]
    with pytest.raises(InvalidInputError, match="run from node edge_u"):
        check_geometry(graph, _geometry(edge_coords=reversed_e0))


def test_straight_geometry_draws_each_edge_from_u_to_v():
    graph = small_graph()
    geometry = straight_geometry(graph, NODE_LON, NODE_LAT)
    assert geometry.edge_offsets.tolist() == [0, 2, 4, 6, 8]
    check_geometry(graph, geometry)
    assert geometry.edge_coords[6:].tolist() == [
        [NODE_LON[1], NODE_LAT[1]],
        [NODE_LON[4], NODE_LAT[4]],
    ]
