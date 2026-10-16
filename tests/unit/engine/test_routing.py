from dataclasses import replace

import numpy as np
import pytest

from evaku_ai.core.errors import InvalidInputError
from evaku_ai.domain.enums import RouteMode, Vehicle
from evaku_ai.domain.rules import RouteRules
from evaku_ai.engine.accessibility import NO_NODE
from evaku_ai.engine.flooding import depth_from_water_level, edge_status, passable_edges
from evaku_ai.engine.graph import RoadGraph
from evaku_ai.engine.routing import (
    edge_lookup,
    path_edges,
    path_from_predecessors,
    plan_routes,
    route_coordinates,
)
from tests.fixtures.small_graph import (
    E3_BEND,
    NODE_LAT,
    NODE_LON,
    small_geometry,
    small_population,
    small_scenario,
)

VILLAGE_A, VILLAGE_B = 0, 1


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


def _status(scenario, water_level_m):
    depth = depth_from_water_level(scenario.graph.hand_min, water_level_m)
    return edge_status(scenario.graph, depth)[0]


def _route(water_level_m, vehicle=Vehicle.CAR, village=VILLAGE_B, hubs=None, **changes):
    scenario = small_scenario(**changes)
    status = _status(scenario, water_level_m)
    hub_nodes = scenario.hub_nodes if hubs is None else hubs
    [route] = plan_routes(scenario, small_geometry(), status, vehicle, [village], hub_nodes)
    return route


def _node(i):
    return (float(NODE_LON[i]), float(NODE_LAT[i]))


def test_reachable_village_gets_a_land_route_along_the_road_shape():
    route = _route(0.0, village=VILLAGE_A)
    assert route.mode is RouteMode.LAND and route.boat_point is None
    assert (route.village_id, route.hub_id) == ("A", 10)
    assert route.edge_ids == (100, 103)
    assert route.distance_m == pytest.approx(2500.0)
    assert route.minutes == pytest.approx(2.5)
    assert route.coordinates == (_node(0), _node(1), E3_BEND, _node(4))


def test_cut_village_gets_a_boat_leg_from_the_nearest_reachable_node():
    route = _route(1.0)
    assert route.mode is RouteMode.LAND_BOAT
    assert route.edge_ids == (100,)
    assert route.coordinates == (_node(0), _node(1))
    assert route.boat_point is not None
    assert (route.boat_point.lon, route.boat_point.lat) == _node(1)
    assert route.boat_point.water_distance_m == pytest.approx(1000.0)
    assert route.distance_m == pytest.approx(2000.0)
    assert route.minutes == pytest.approx(1 + 1000 / (10 * 1000 / 60))


def test_limited_road_is_land_for_trucks_and_boat_for_cars():
    truck = _route(0.9, Vehicle.TRUCK)
    assert truck.mode is RouteMode.LAND and truck.edge_ids == (100, 101)
    assert _route(0.9, Vehicle.CAR).mode is RouteMode.LAND_BOAT


@pytest.mark.parametrize("water_level_m", [0.0, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("vehicle", list(Vehicle))
def test_routes_never_use_impassable_roads(water_level_m, vehicle):
    scenario = small_scenario()
    status = _status(scenario, water_level_m)
    usable = set(scenario.edge_ids[passable_edges(status, vehicle)].tolist())
    routes = plan_routes(
        scenario, small_geometry(), status, vehicle, [VILLAGE_A, VILLAGE_B], scenario.hub_nodes
    )
    for route in routes:
        assert set(route.edge_ids) <= usable


def test_edges_walked_against_their_direction_are_reversed():
    route = _route(0.0, hubs={11: 4})
    assert route.edge_ids == (103, 101)
    assert route.coordinates == (_node(4), E3_BEND, _node(1), _node(2))


def test_fastest_hub_is_chosen_and_reported():
    route = _route(0.0, hubs={10: 0, 11: 4})
    assert route.hub_id == 10
    assert route.minutes == pytest.approx(2.0)


def test_village_on_a_hub_node_has_a_zero_length_route():
    route = _route(0.0, village=VILLAGE_A, hubs={11: 4, 12: 4})
    assert route.hub_id == 11
    assert route.mode is RouteMode.LAND
    assert route.coordinates == (_node(4), _node(4))
    assert (route.distance_m, route.minutes, route.edge_ids) == (0.0, 0.0, ())


def test_village_without_road_node_is_rejected():
    population = replace(small_population(), village=np.array([NO_NODE] * 4 + [0]))
    with pytest.raises(InvalidInputError, match="B has no road node"):
        _route(0.0, population=population)


@pytest.mark.parametrize(
    ("status", "villages", "hubs", "message"),
    [
        (np.zeros(3, dtype=np.int8), [0], {10: 0}, "status must have length"),
        (np.zeros(4, dtype=np.int8), [0], {}, "at least one hub"),
        (np.zeros(4, dtype=np.int8), [0], {10: 9}, "hub nodes"),
        (np.zeros(4, dtype=np.int8), [2], {10: 0}, "village index"),
    ],
)
def test_invalid_inputs_are_rejected(status, villages, hubs, message):
    with pytest.raises(InvalidInputError, match=message):
        plan_routes(small_scenario(), small_geometry(), status, Vehicle.CAR, villages, hubs)


def test_route_coordinates_reject_an_edge_that_misses_the_node():
    scenario = small_scenario()
    with pytest.raises(InvalidInputError, match="does not start or end"):
        route_coordinates(scenario.graph, small_geometry(), [0, 2], [1])


def test_boat_speed_changes_the_minutes():
    scenario = small_scenario()
    [route] = plan_routes(
        scenario,
        small_geometry(),
        _status(scenario, 1.0),
        Vehicle.CAR,
        [VILLAGE_B],
        scenario.hub_nodes,
        RouteRules(boat_speed_kmh=5.0),
    )
    assert route.minutes == pytest.approx(13.0)


def test_routes_keep_the_order_of_the_requested_villages():
    scenario = small_scenario()
    status = _status(scenario, 1.0)
    routes = plan_routes(
        scenario,
        small_geometry(),
        status,
        Vehicle.CAR,
        [VILLAGE_B, VILLAGE_A, VILLAGE_B],
        scenario.hub_nodes,
    )
    assert [r.village_id for r in routes] == ["B", "A", "B"]
    assert routes[0] == routes[2]
