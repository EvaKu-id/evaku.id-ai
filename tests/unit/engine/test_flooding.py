from dataclasses import replace

import numpy as np
import pytest

from evaku_ai.core.errors import InvalidInputError
from evaku_ai.domain.enums import RoadCondition, Vehicle
from evaku_ai.domain.rules import PassabilityThresholds, passable, road_condition
from evaku_ai.engine.flooding import (
    bridge_approach_depth,
    depth_from_observation,
    depth_from_water_level,
    edge_status,
    passable_edges,
    road_status,
)
from evaku_ai.engine.graph import RoadGraph
from tests.fixtures.small_graph import small_graph

SAFE, LIMITED, CUT, UNKNOWN = (int(c) for c in RoadCondition)


def test_depth_from_water_level():
    depth = depth_from_water_level(small_graph().hand_min, water_level_m=1.0)
    np.testing.assert_array_equal(depth, [0, 50, 0, 0])


def test_zero_water_means_every_road_is_safe():
    graph = small_graph()
    depth = bridge_approach_depth(graph, depth_from_water_level(graph.hand_min, 0.0))
    np.testing.assert_array_equal(road_status(depth), [SAFE] * graph.n_edges)


def test_disconnected_edges_stay_dry():
    connected = np.array([True, False, True, True])
    depth = depth_from_water_level(small_graph().hand_min, 1.0, connected=connected)
    np.testing.assert_array_equal(depth, [0, 0, 0, 0])


def test_unknown_hand_gives_unknown_depth():
    depth = depth_from_water_level(np.array([np.nan, 0.5]), 1.0)
    assert np.isnan(depth[0])
    assert depth[1] == 50


@pytest.mark.parametrize("level", [-0.1, np.nan, np.inf])
def test_invalid_water_level(level):
    with pytest.raises(InvalidInputError, match="water_level_m"):
        depth_from_water_level(small_graph().hand_min, level)


def test_connected_mask_must_match():
    with pytest.raises(InvalidInputError, match="connected"):
        depth_from_water_level(small_graph().hand_min, 1.0, connected=np.array([True]))


def test_road_status_boundaries():
    depth = np.array([0, 30, 31, 60, 61, np.nan])
    status = road_status(depth)
    assert status.dtype == np.int8
    np.testing.assert_array_equal(status, [SAFE, SAFE, LIMITED, LIMITED, CUT, UNKNOWN])


def test_road_status_matches_domain_rule():
    depth = np.arange(0, 201, dtype=float)
    expected = [road_condition(d) for d in depth]
    np.testing.assert_array_equal(road_status(depth), expected)


def test_road_status_uses_given_thresholds():
    status = road_status(np.array([25.0, 45.0]), PassabilityThresholds(car_cm=20, truck_cm=40))
    np.testing.assert_array_equal(status, [LIMITED, CUT])


def test_bridge_judged_by_approach_not_river():
    depth = bridge_approach_depth(small_graph(), np.array([0.0, 50.0, 500.0, 0.0]))
    np.testing.assert_array_equal(depth, [0, 50, 50, 0])
    assert road_status(depth)[2] == LIMITED


def test_bridge_segments_share_both_approaches():
    graph = RoadGraph(
        edge_u=np.array([0, 1, 2, 3]),
        edge_v=np.array([1, 2, 3, 4]),
        travel_s=np.array([10.0, 10.0, 10.0, 10.0]),
        length_m=np.array([100.0, 50.0, 50.0, 100.0]),
        hand_min=np.array([3.0, 0.0, 0.0, 3.0]),
        is_bridge=np.array([False, True, True, False]),
        node_x=np.arange(5, dtype=float),
        node_y=np.zeros(5),
    )
    depth = bridge_approach_depth(graph, np.array([10.0, 400.0, 400.0, 70.0]))
    np.testing.assert_array_equal(depth, [10, 70, 70, 70])


def test_bridge_ignores_unknown_approach_when_another_is_known():
    graph = small_graph()
    depth = bridge_approach_depth(graph, np.array([0.0, 20.0, 500.0, np.nan]))
    assert depth[2] == 20


def test_bridge_without_known_approach_is_unknown():
    graph = small_graph()
    depth = bridge_approach_depth(graph, np.array([0.0, np.nan, 500.0, 0.0]))
    assert np.isnan(depth[2])
    assert road_status(depth)[2] == UNKNOWN


def test_bridge_approach_does_not_mutate_input():
    original = np.array([0.0, 50.0, 500.0, 0.0])
    bridge_approach_depth(small_graph(), original)
    np.testing.assert_array_equal(original, [0, 50, 500, 0])


def test_no_bridges_returns_same_depths():
    graph = replace(small_graph(), is_bridge=np.zeros(4, dtype=bool))
    depth = np.array([0.0, 50.0, 500.0, 0.0])
    np.testing.assert_array_equal(bridge_approach_depth(graph, depth), depth)


def test_bridge_depth_length_must_match():
    with pytest.raises(InvalidInputError, match="depth_cm"):
        bridge_approach_depth(small_graph(), np.array([0.0]))


@pytest.mark.parametrize("vehicle", list(Vehicle))
def test_passable_edges_follow_domain_rule(vehicle):
    status = np.array([SAFE, LIMITED, CUT, UNKNOWN], dtype=np.int8)
    expected = [passable(RoadCondition(code), vehicle) for code in status]
    np.testing.assert_array_equal(passable_edges(status, vehicle), expected)


def test_passable_edges_empty():
    assert passable_edges(np.array([], dtype=np.int8), Vehicle.CAR).shape == (0,)


@pytest.mark.parametrize(
    "status",
    [
        np.array([0, 4], dtype=np.int8),
        np.array([-1], dtype=np.int8),
        np.zeros((2, 2), dtype=np.int8),
    ],
)
def test_passable_edges_rejects_bad_status(status):
    with pytest.raises(InvalidInputError, match="status"):
        passable_edges(status, Vehicle.TRUCK)


def test_depth_from_observation_converts_and_keeps_unknown():
    depth = depth_from_observation(np.array([0.0, 0.3, 0.61, np.nan]))
    np.testing.assert_allclose(depth[:3], [0.0, 30.0, 61.0])
    assert np.isnan(depth[3])
    assert depth_from_observation(np.array([np.nan])).shape == (1,)


@pytest.mark.parametrize("bad", [-0.1, np.inf, -np.inf])
def test_depth_from_observation_rejects_bad_values(bad):
    with pytest.raises(InvalidInputError, match="observed depth"):
        depth_from_observation(np.array([0.2, bad]))


@pytest.mark.parametrize(
    ("depth_cm", "expected"),
    [(30.0, SAFE), (30.5, LIMITED), (60.0, LIMITED), (60.5, CUT), (np.nan, UNKNOWN)],
)
def test_edge_status_judges_bridges_by_their_approach(depth_cm, expected):
    graph = small_graph()
    depth = np.full(graph.n_edges, depth_cm)
    depth[graph.is_bridge] = 500.0
    status, effective = edge_status(graph, depth)
    assert (status == expected).all()
    np.testing.assert_array_equal(effective, bridge_approach_depth(graph, depth))
