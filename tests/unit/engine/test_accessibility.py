from dataclasses import replace

import numpy as np
import pytest

from evaku_ai.core.errors import InvalidInputError
from evaku_ai.domain.enums import NodeImpact, RoadCondition, Vehicle, VillageAccessStatus
from evaku_ai.engine.accessibility import (
    NO_NODE,
    NodePopulation,
    access_difficulty,
    access_times,
    classify_villages,
    facility_access,
    health_access_gap,
    node_impact,
    travel_time_from_hubs,
)
from evaku_ai.engine.flooding import bridge_approach_depth, depth_from_water_level, road_status
from evaku_ai.engine.graph import RoadGraph, to_csr
from tests.fixtures.small_graph import HUB_NODES, VILLAGE_IDS, small_graph, small_population

SAFE, LIMITED, CUT, UNKNOWN = (int(c) for c in RoadCondition)
NORMAL_TIME = [0.0, 60.0, 120.0, 150.0, 150.0]


def status_at(water_level_m):
    graph = small_graph()
    depth = depth_from_water_level(graph.hand_min, water_level_m)
    return road_status(bridge_approach_depth(graph, depth))


def test_normal_graph_times_hubs_and_predecessors():
    reach = travel_time_from_hubs(to_csr(small_graph()), HUB_NODES)
    np.testing.assert_array_equal(reach.time_s, NORMAL_TIME)
    np.testing.assert_array_equal(reach.hub, [0, 0, 0, 0, 0])
    np.testing.assert_array_equal(reach.predecessor, [NO_NODE, 0, 1, 2, 1])
    assert reach.reachable.all()


def test_dry_status_matches_normal_for_every_vehicle():
    times = access_times(small_graph(), status_at(0.0), HUB_NODES)
    for vehicle in Vehicle:
        np.testing.assert_array_equal(times.flood[vehicle].time_s, times.normal.time_s)


def test_one_metre_cuts_car_access_but_not_truck():
    times = access_times(small_graph(), status_at(1.0), HUB_NODES)
    car = times.flood[Vehicle.CAR]
    np.testing.assert_array_equal(car.reachable, [True, True, False, False, True])
    np.testing.assert_array_equal(car.hub[2:4], [NO_NODE, NO_NODE])
    np.testing.assert_array_equal(car.predecessor[2:4], [NO_NODE, NO_NODE])
    np.testing.assert_array_equal(times.flood[Vehicle.TRUCK].time_s, NORMAL_TIME)
    np.testing.assert_array_equal(times.normal.time_s, NORMAL_TIME)


def test_two_metres_cuts_truck_access_too():
    times = access_times(small_graph(), status_at(2.0), HUB_NODES)
    for vehicle in Vehicle:
        np.testing.assert_array_equal(
            times.flood[vehicle].reachable, [True, True, False, False, True]
        )


def test_cut_bridge_isolates_only_the_far_side():
    status = np.array([SAFE, SAFE, CUT, SAFE], dtype=np.int8)
    reach = access_times(small_graph(), status, HUB_NODES).flood[Vehicle.TRUCK]
    np.testing.assert_array_equal(reach.reachable, [True, True, True, False, True])
    assert np.isinf(reach.time_s[3])


def test_unknown_roads_are_traversed():
    status = np.array([SAFE, UNKNOWN, UNKNOWN, SAFE], dtype=np.int8)
    reach = access_times(small_graph(), status, HUB_NODES).flood[Vehicle.CAR]
    np.testing.assert_array_equal(reach.time_s, NORMAL_TIME)


def test_nearest_hub_wins():
    reach = travel_time_from_hubs(to_csr(small_graph()), np.array([0, 3]))
    np.testing.assert_array_equal(reach.time_s, [0.0, 60.0, 30.0, 0.0, 150.0])
    np.testing.assert_array_equal(reach.hub, [0, 0, 3, 3, 0])
    np.testing.assert_array_equal(reach.predecessor, [NO_NODE, 0, 3, NO_NODE, 1])


def test_duplicate_hubs_are_accepted():
    reach = travel_time_from_hubs(to_csr(small_graph()), np.array([0, 0]))
    np.testing.assert_array_equal(reach.time_s, NORMAL_TIME)


def test_isolated_hub_reaches_only_itself():
    status = np.array([CUT, CUT, CUT, CUT], dtype=np.int8)
    reach = access_times(small_graph(), status, HUB_NODES).flood[Vehicle.TRUCK]
    np.testing.assert_array_equal(reach.reachable, [True, False, False, False, False])
    assert reach.time_s[0] == 0.0


@pytest.mark.parametrize(
    ("hubs", "message"),
    [
        (np.array([], dtype=int), "non-empty"),
        (np.array([0.0]), "integer"),
        (np.array([[0]]), "1-D"),
        (np.array([5]), "out of range"),
        (np.array([-1]), "out of range"),
    ],
)
def test_rejects_invalid_hubs(hubs, message):
    with pytest.raises(InvalidInputError, match=message):
        travel_time_from_hubs(to_csr(small_graph()), hubs)


def test_rejects_status_of_wrong_length():
    with pytest.raises(InvalidInputError, match="status"):
        access_times(small_graph(), np.array([SAFE, SAFE], dtype=np.int8), HUB_NODES)


REACHABLE = VillageAccessStatus.REACHABLE
PARTIAL = VillageAccessStatus.PARTIALLY_ISOLATED
FULL = VillageAccessStatus.FULLY_ISOLATED
BRIDGE_CUT = np.array([SAFE, SAFE, CUT, SAFE], dtype=np.int8)
NOT_FLOODED = np.zeros(5, dtype=bool)


def classify(status, population=None, flooded=NOT_FLOODED, village_ids=VILLAGE_IDS):
    access = access_times(small_graph(), status, HUB_NODES)
    return classify_villages(access, population or small_population(), flooded, village_ids)


def with_totals(**nodes):
    total = small_population().total.copy()
    for node, count in nodes.items():
        total[int(node[1:])] = count
    return replace(small_population(), total=total)


def test_dry_roads_leave_every_village_reachable():
    villages = classify(status_at(0.0))
    assert [v.status for v in villages] == [REACHABLE, REACHABLE]
    assert all(v.isolated_population == 0 for v in villages)


def test_truck_only_access_makes_village_partially_isolated():
    village_a, village_b = classify(status_at(1.0))
    assert village_a.status is REACHABLE
    assert village_b.status is PARTIAL
    assert village_b.isolated_population == 0


def test_no_land_access_makes_village_fully_isolated():
    village_a, village_b = classify(status_at(2.0))
    assert village_a.status is REACHABLE
    assert village_b.status is FULL
    assert village_b.village_id == "B"
    assert (village_b.isolated_population, village_b.isolated_under5) == (200, 22)
    assert village_b.isolated_elderly == 17
    assert village_b.priority_score is None


def test_half_cut_village_counts_people_exactly():
    village_b = classify(BRIDGE_CUT)[1]
    assert village_b.status is PARTIAL
    assert (village_b.isolated_population, village_b.isolated_under5) == (100, 12)


def test_majority_cut_is_fully_isolated():
    village_b = classify(BRIDGE_CUT, with_totals(n2=49.0, n3=51.0))[1]
    assert village_b.status is FULL
    assert village_b.isolated_population == 51


@pytest.mark.parametrize(("cut_people", "expected"), [(9.0, REACHABLE), (10.0, PARTIAL)])
def test_partial_threshold_is_ten_percent(cut_people, expected):
    population = with_totals(n2=100.0 - cut_people, n3=cut_people)
    assert classify(BRIDGE_CUT, population)[1].status is expected


@pytest.mark.parametrize(("detour_leg_s", "expected"), [(100.0, PARTIAL), (90.0, REACHABLE)])
def test_detour_longer_than_three_times_normal_is_partial(detour_leg_s, expected):
    access = detour_access(detour_leg_s)
    population = NodePopulation(
        village=np.array([NO_NODE, 0, NO_NODE]),
        total=np.array([0.0, 40.0, 0.0]),
        under5=np.zeros(3),
        elderly=np.zeros(3),
    )
    (village,) = classify_villages(access, population, np.zeros(3, dtype=bool), ["V"])
    assert village.status is expected
    assert village.isolated_population == 0


def detour_access(detour_leg_s):
    graph = RoadGraph(
        edge_u=np.array([0, 0, 2]),
        edge_v=np.array([1, 2, 1]),
        travel_s=np.array([60.0, detour_leg_s, detour_leg_s]),
        length_m=np.array([1000.0, 1500.0, 1500.0]),
        hand_min=np.array([5.0, 5.0, 5.0]),
        is_bridge=np.array([False, False, False]),
        node_x=np.array([0.0, 1000.0, 500.0]),
        node_y=np.array([0.0, 0.0, 800.0]),
    )
    return access_times(graph, np.array([CUT, SAFE, SAFE], dtype=np.int8), HUB_NODES)


def test_nodes_unreachable_on_dry_roads_are_not_flood_isolated():
    base = small_graph()
    graph = replace(base, node_x=np.append(base.node_x, 0.0), node_y=np.append(base.node_y, 0.0))
    population = NodePopulation(
        village=np.array([NO_NODE, NO_NODE, 1, 1, 0, 1]),
        total=np.array([0.0, 0.0, 100.0, 100.0, 50.0, 300.0]),
        under5=np.zeros(6),
        elderly=np.zeros(6),
    )
    access = access_times(graph, status_at(0.0), HUB_NODES)
    village_b = classify_villages(access, population, np.zeros(6, dtype=bool), VILLAGE_IDS)[1]
    assert village_b.status is REACHABLE
    assert village_b.isolated_population == 0


def test_people_outside_villages_and_empty_villages():
    population = with_totals(n0=1000.0, n1=500.0)
    villages = classify(status_at(2.0), population, village_ids=["A", "B", "C"])
    assert [v.isolated_population for v in villages] == [0, 200, 0]
    assert villages[2].status is REACHABLE


def test_flooded_population_follows_mask():
    flooded = np.array([False, False, True, False, True])
    village_a, village_b = classify(status_at(0.0), flooded=flooded)
    assert (village_a.flooded_population, village_b.flooded_population) == (50, 100)


def test_classify_rejects_mismatched_inputs():
    short = NodePopulation(
        village=np.array([0, 0]), total=np.ones(2), under5=np.ones(2), elderly=np.ones(2)
    )
    with pytest.raises(InvalidInputError, match="length"):
        classify(status_at(0.0), short)
    with pytest.raises(InvalidInputError, match="out of range"):
        classify(status_at(0.0), village_ids=["A"])
    with pytest.raises(InvalidInputError, match="flooded_node"):
        classify(status_at(0.0), flooded=np.zeros(5))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"total": np.array([0.0, 0.0, -1.0, 100.0, 50.0])}, "finite"),
        ({"under5": np.array([0.0, 0.0, np.nan, 12.0, 5.0])}, "finite"),
        ({"elderly": np.zeros(4)}, "same length"),
        ({"village": np.array([-1.0, -1.0, 1.0, 1.0, 0.0])}, "integer"),
        ({"village": np.array([-2, -1, 1, 1, 0])}, ">="),
    ],
)
def test_population_rejects_invalid_arrays(changes, message):
    with pytest.raises(InvalidInputError, match=message):
        replace(small_population(), **changes)


def test_facility_access_reports_flooding_and_reach():
    access = access_times(small_graph(), BRIDGE_CUT, HUB_NODES)
    facilities = facility_access(access, np.array([3, 4]), np.array([True, False]))
    np.testing.assert_array_equal(facilities.flooded, [True, False])
    for vehicle in Vehicle:
        np.testing.assert_array_equal(facilities.reachable[vehicle], [False, True])


def test_facility_access_accepts_no_facilities():
    access = access_times(small_graph(), BRIDGE_CUT, HUB_NODES)
    facilities = facility_access(access, np.array([], dtype=int), np.array([], dtype=bool))
    assert facilities.reachable[Vehicle.CAR].shape == (0,)


@pytest.mark.parametrize(
    ("nodes", "flooded", "message"),
    [
        (np.array([9]), np.array([False]), "out of range"),
        (np.array([1, 2]), np.array([False]), "facility_flooded"),
    ],
)
def test_facility_access_rejects_invalid_inputs(nodes, flooded, message):
    access = access_times(small_graph(), BRIDGE_CUT, HUB_NODES)
    with pytest.raises(InvalidInputError, match=message):
        facility_access(access, nodes, flooded)


@pytest.mark.parametrize(
    ("water_level_m", "village_b_impact"),
    [(0.0, NodeImpact.NORMAL), (1.0, NodeImpact.TRUCK_ONLY), (2.0, NodeImpact.CUT)],
)
def test_node_impact_by_water_level(water_level_m, village_b_impact):
    access = access_times(small_graph(), status_at(water_level_m), HUB_NODES)
    impact = node_impact(access)
    np.testing.assert_array_equal(impact[[0, 1, 4]], [NodeImpact.NORMAL] * 3)
    np.testing.assert_array_equal(impact[[2, 3]], [village_b_impact] * 2)


def test_node_impact_marks_long_detours():
    assert node_impact(detour_access(100.0))[1] == NodeImpact.DELAYED


@pytest.mark.parametrize(
    ("status", "village_b"),
    [(status_at(0.0), 0.0), (status_at(1.0), 2 / 3), (status_at(2.0), 1.0), (BRIDGE_CUT, 0.5)],
)
def test_access_difficulty_is_population_weighted(status, village_b):
    access = access_times(small_graph(), status, HUB_NODES)
    difficulty = access_difficulty(access, small_population(), 3)
    np.testing.assert_allclose(difficulty, [0.0, village_b, 0.0])


HEALTH_NODES = np.array([4])


def health_gap(flooded, nodes=HEALTH_NODES, status=BRIDGE_CUT):
    return health_access_gap(small_graph(), status, nodes, flooded, small_population(), 3)


def test_health_gap_counts_people_cut_from_working_facilities():
    np.testing.assert_allclose(health_gap(np.array([False])), [0.0, 0.5, 0.0])


def test_health_gap_is_full_without_working_facility():
    np.testing.assert_allclose(health_gap(np.array([True])), [1.0, 1.0, 0.0])
    empty = health_gap(np.array([], dtype=bool), nodes=np.array([], dtype=int))
    np.testing.assert_allclose(empty, [1.0, 1.0, 0.0])


@pytest.mark.parametrize(
    ("nodes", "flooded", "message"),
    [
        (np.array([7]), np.array([False]), "out of range"),
        (np.array([4]), np.array([0]), "facility_flooded"),
    ],
)
def test_health_gap_rejects_invalid_inputs(nodes, flooded, message):
    with pytest.raises(InvalidInputError, match=message):
        health_gap(flooded, nodes=nodes)


def test_component_helpers_check_population():
    access = access_times(small_graph(), BRIDGE_CUT, HUB_NODES)
    with pytest.raises(InvalidInputError, match="out of range"):
        access_difficulty(access, small_population(), 1)
