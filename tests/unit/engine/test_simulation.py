import numpy as np
import pytest

from evaku_ai.core.errors import InvalidInputError
from evaku_ai.domain.enums import RoadCondition, Vehicle, VillageAccessStatus
from evaku_ai.engine.flooding import depth_from_water_level, edge_status
from evaku_ai.engine.simulation import run_simulation
from tests.fixtures.small_graph import small_scenario

REACHABLE = VillageAccessStatus.REACHABLE
PARTIAL = VillageAccessStatus.PARTIALLY_ISOLATED
FULL = VillageAccessStatus.FULLY_ISOLATED


def simulate(level, vehicle=Vehicle.CAR, hub_ids=None, river_id="r1", **changes):
    return run_simulation(small_scenario(**changes), river_id, level, vehicle, hub_ids)


def test_dry_scenario_changes_nothing():
    result = simulate(0.0)
    assert result.roads == []
    assert [v.status for v in result.villages] == [REACHABLE, REACHABLE]
    assert [v.priority_score for v in result.villages] == [0.0, 0.0]
    assert (result.summary.isolated_villages, result.summary.isolated_population) == (0, 0)
    assert result.summary.cut_road_km == 0.0


def test_one_metre_limits_roads_for_cars():
    result = simulate(1.0)
    assert [(r.edge_id, r.status, r.depth_cm) for r in result.roads] == [
        (101, RoadCondition.LIMITED, 50),
        (102, RoadCondition.LIMITED, 50),
    ]
    village_a, village_b = result.villages
    assert (village_a.status, village_b.status) == (REACHABLE, PARTIAL)
    assert village_b.priority_score == pytest.approx(100.0 * 0.10 / 0.85)
    assert result.summary.isolated_villages == 1
    assert result.summary.cut_road_km == pytest.approx(1.5)


def test_cut_road_km_follows_vehicle():
    assert simulate(1.0, Vehicle.TRUCK).summary.cut_road_km == 0.0
    assert simulate(2.0, Vehicle.TRUCK).summary.cut_road_km == pytest.approx(1.5)


def test_two_metres_isolates_village_b():
    result = simulate(2.0)
    village_b = result.villages[1]
    assert village_b.status is FULL
    assert village_b.isolated_population == 200
    assert village_b.priority_score == pytest.approx(100.0)
    assert result.roads[0].depth_cm == 150
    assert (result.summary.isolated_villages, result.summary.isolated_population) == (1, 200)


def test_breakdown_explains_village_score():
    result = simulate(2.0)
    breakdown = result.scores.breakdown(1)
    assert breakdown["isolation_duration"]["weight"] == 0.0
    assert breakdown["isolated_population"]["raw"] == 200.0


def test_roads_on_another_river_stay_dry():
    result = simulate(2.0, edge_river=np.array([0, 1, 0, 0]))
    assert result.roads == []
    assert result.villages[1].status is REACHABLE


def test_flooded_population_follows_node_hand():
    village_a, village_b = simulate(1.0).villages
    assert (village_a.flooded_population, village_b.flooded_population) == (0, 100)


def test_flooded_health_facility_counts_as_missing():
    result = simulate(1.0, health_hand=np.array([0.5]))
    np.testing.assert_array_equal(result.scores.raw["no_health_facility"], [1.0, 1.0])


def test_hub_selection():
    hubs = {10: 0, 11: 3}
    assert simulate(2.0, hub_nodes=hubs).villages[1].status is PARTIAL
    assert simulate(2.0, hub_ids=[10], hub_nodes=hubs).villages[1].status is FULL


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"river_id": "r9"}, "river_id"), ({"hub_ids": [99]}, "hub_ids")],
)
def test_rejects_unknown_ids(kwargs, message):
    with pytest.raises(InvalidInputError, match=message):
        simulate(1.0, **kwargs)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"edge_ids": np.array([100, 101, 102])}, "edge_ids"),
        ({"edge_ids": np.array([100, 100, 102, 103])}, "unique"),
        ({"edge_river": np.array([0, 2, 0, 0])}, "out of range"),
        ({"node_river": np.zeros(5)}, "integer"),
        ({"node_hand": np.zeros(4)}, "node_hand"),
        ({"health_nodes": np.array([5])}, "health_nodes"),
        ({"health_hand": np.zeros(2)}, "health_hand"),
        ({"hub_nodes": {}}, "hub"),
        ({"hub_nodes": {10: 9}}, "hub_nodes"),
        ({"village_ids": ["A"]}, "village"),
    ],
)
def test_scenario_data_rejects_invalid_arrays(changes, message):
    with pytest.raises(InvalidInputError, match=message):
        small_scenario(**changes)


def test_result_keeps_edge_status_and_hubs_for_routing():
    scenario = small_scenario(hub_nodes={10: 0, 11: 3})
    result = run_simulation(scenario, "r1", 1.0, Vehicle.CAR, [11, 11])
    expected, _ = edge_status(scenario.graph, depth_from_water_level(scenario.graph.hand_min, 1.0))
    np.testing.assert_array_equal(result.edge_status, expected)
    assert result.hub_nodes == {11: 3}
    assert run_simulation(scenario, "r1", 1.0, Vehicle.CAR).hub_nodes == {10: 0, 11: 3}
