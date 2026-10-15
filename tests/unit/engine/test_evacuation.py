import numpy as np
import pytest

from evaku_ai.core.errors import InvalidInputError
from evaku_ai.domain.enums import FacilityType
from evaku_ai.domain.rules import EvacuationRules
from evaku_ai.engine.accessibility import NO_NODE, NodePopulation
from evaku_ai.engine.evacuation import (
    EvacuationSites,
    Horizon,
    nearest_safe_sites,
    plan_evacuation,
    safe_sites,
    village_origins,
)
from evaku_ai.engine.graph import RoadGraph
from evaku_ai.engine.simulation import ScenarioData

# Nodes: 0-1-2-3 along a road, 0-4 a short spur, 4-5 a very long road, 1-6 a short spur.
# Edge 2 (2-3) sits low and floods at 2 m; every other edge stays dry.
EDGES = [
    (0, 1, 1000.0),
    (1, 2, 1000.0),
    (2, 3, 1000.0),
    (0, 4, 500.0),
    (4, 5, 12000.0),
    (1, 6, 300.0),
]
VILLAGES = ["V0", "V1", "V2"]
SITE_IDS = {"f10": 10, "f11": 11, "f12": 12, "f13": 13, "f14": 14}


def _scenario(edges=EDGES):
    u, v, length = (np.array(column) for column in zip(*edges, strict=True))
    hand_min = np.full(len(edges), 10.0)
    hand_min[2] = 1.0
    graph = RoadGraph(
        edge_u=u.astype(np.int64),
        edge_v=v.astype(np.int64),
        travel_s=length / 10.0,
        length_m=length.astype(float),
        hand_min=hand_min,
        is_bridge=np.zeros(len(edges), dtype=bool),
        node_x=np.arange(7, dtype=float),
        node_y=np.zeros(7),
    )
    village = np.array([0, 0, 1, NO_NODE, NO_NODE, NO_NODE, NO_NODE])
    total = np.array([50.0, 10.0, 30.0, 0, 0, 0, 0])
    return ScenarioData(
        graph=graph,
        edge_ids=np.arange(100, 100 + len(edges), dtype=np.int64),
        rivers=["r1"],
        edge_river=np.zeros(len(edges), dtype=np.int64),
        node_river=np.zeros(7, dtype=np.int64),
        node_hand=np.full(7, 10.0),
        population=NodePopulation(village, total, total * 0.1, total * 0.1),
        village_ids=VILLAGES,
        hub_nodes={1: 0},
        health_nodes=np.empty(0, dtype=np.int64),
        health_hand=np.empty(0),
        health_river=np.empty(0, dtype=np.int64),
    )


def _sites():
    return EvacuationSites(
        facility_ids=np.array([10, 11, 12, 13, 14]),
        nodes=np.array([3, 6, 5, 4, 3]),
        hand=np.array([5.0, 1.5, 10.0, 3.0, 4.0]),
        types=["school", "place_of_worship", "field", "school", "village_office"],
        preferred=np.array([False, False, False, False, True]),
    )


def _horizons(eligible=(True, True, True)):
    mask = np.array(eligible)
    return [Horizon(24, 0.5, mask), Horizon(48, 2.0, mask), Horizon(72, 0.5, mask)]


def _by_village(plan):
    rows: dict[str, list] = {}
    for row in plan.sites:
        rows.setdefault(row.village_id, []).append(row)
    return rows


def test_village_origins_pick_the_most_populated_node():
    population = NodePopulation(
        village=np.array([0, 0, 1, 1, NO_NODE]),
        total=np.array([5.0, 9.0, 4.0, 4.0, 99.0]),
        under5=np.zeros(5),
        elderly=np.zeros(5),
    )
    assert village_origins(population, 3).tolist() == [1, 2, NO_NODE]
    empty = NodePopulation(np.full(2, NO_NODE), np.ones(2), np.zeros(2), np.zeros(2))
    assert village_origins(empty, 1).tolist() == [NO_NODE]


def test_safe_sites_need_margin_and_allowed_type():
    sites = _sites()
    assert safe_sites(sites, 0.5, EvacuationRules()).tolist() == [True] * 5
    assert safe_sites(sites, 2.0, EvacuationRules()).tolist() == [True, False, True, True, True]
    nan_sites = EvacuationSites(
        np.array([1]), np.array([0]), np.array([np.nan]), ["school"], np.array([False])
    )
    assert not safe_sites(nan_sites, 0.0, EvacuationRules()).any()
    schools = EvacuationRules(site_types=(FacilityType.SCHOOL,))
    assert safe_sites(sites, 0.5, schools).tolist() == [True, False, False, True, False]


def test_nearest_safe_sites_order_preferred_first_then_distance():
    distance = np.array([0.0, 1000.0, 2000.0, 3000.0, 500.0, 12500.0, 1300.0])
    chosen = nearest_safe_sites(distance, _sites(), np.ones(5, dtype=bool), EvacuationRules())
    assert chosen.tolist() == [4, 3, 1]
    nearest_two = nearest_safe_sites(
        distance, _sites(), np.ones(5, dtype=bool), EvacuationRules(max_sites=2)
    )
    assert nearest_two.tolist() == [4, 3]


def test_plan_picks_three_safe_sites_with_routes_and_times():
    plans = plan_evacuation(_scenario(), _sites(), "r1", _horizons())
    assert [p.horizon_hours for p in plans] == [24, 48, 72]
    first = _by_village(plans[0])
    v0 = first["V0"]
    assert [r.facility_id for r in v0] == [14, 13, 11]
    assert [r.rank for r in v0] == [1, 2, 3]
    to_f11 = v0[2]
    assert to_f11.distance_m == 1300.0
    assert to_f11.route_edge_ids == (100, 105)
    assert to_f11.walk_minutes == pytest.approx(1300 / (4000 / 60))
    assert to_f11.drive_minutes == pytest.approx(130 / 60)
    assert [r.facility_id for r in first["V1"]] == [14, 10, 11]
    assert plans[0].unserved == ["V2"]


def test_leave_before_hour_is_the_first_horizon_the_site_is_lost():
    v0 = _by_village(plan_evacuation(_scenario(), _sites(), "r1", _horizons())[0])["V0"]
    leave = {r.facility_id: r.leave_before_hour for r in v0}
    assert leave == {14: 48, 13: None, 11: 48}


def test_sites_flooded_at_the_horizon_are_never_chosen():
    plans = plan_evacuation(_scenario(), _sites(), "r1", _horizons())
    second = _by_village(plans[1])
    assert [r.facility_id for r in second["V0"]] == [14, 13, 10]
    assert 11 not in {r.facility_id for r in plans[1].sites}
    assert {r.facility_id: r.leave_before_hour for r in second["V0"]} == {14: 48, 13: None, 10: 48}


def test_villages_cut_off_later_still_get_routes_with_a_deadline():
    v1 = _by_village(plan_evacuation(_scenario(), _sites(), "r1", _horizons())[1])["V1"]
    assert [r.facility_id for r in v1] == [14, 10, 13]
    assert v1[0].route_edge_ids == (102,)
    assert v1[0].leave_before_hour == 48


def test_routes_follow_current_road_conditions():
    plans = plan_evacuation(_scenario(), _sites(), "r1", _horizons(), current_water_level_m=2.0)
    v1 = _by_village(plans[1])["V1"]
    assert [r.facility_id for r in v1] == [13]
    assert v1[0].route_edge_ids == (101, 100, 103)
    for plan in plans:
        for row in plan.sites:
            assert 102 not in row.route_edge_ids


def test_deadline_follows_the_shown_route_even_if_a_detour_exists():
    with_detour = [*EDGES, (1, 3, 4000.0)]
    plans = plan_evacuation(_scenario(with_detour), _sites(), "r1", _horizons())
    v1 = _by_village(plans[0])["V1"]
    assert v1[0].facility_id == 14
    assert v1[0].route_edge_ids == (102,)
    assert v1[0].leave_before_hour == 48


@pytest.mark.parametrize("level", [-1.0, float("nan")])
def test_bad_current_water_level_is_rejected(level):
    with pytest.raises(InvalidInputError, match="current water level"):
        plan_evacuation(_scenario(), _sites(), "r1", _horizons(), current_water_level_m=level)


def test_only_eligible_villages_get_rows():
    plans = plan_evacuation(_scenario(), _sites(), "r1", _horizons((False, True, False)))
    assert {r.village_id for p in plans for r in p.sites} == {"V1"}
    assert all(p.unserved == [] for p in plans)


def test_village_without_safe_site_in_reach_is_unserved():
    rules = EvacuationRules(max_distance_m=400.0)
    plans = plan_evacuation(_scenario(), _sites(), "r1", _horizons(), rules)
    assert plans[0].sites == []
    assert plans[0].unserved == VILLAGES


def test_no_eligible_village_returns_empty_plans():
    plans = plan_evacuation(_scenario(), _sites(), "r1", _horizons((False, False, False)))
    assert all(p.sites == [] and p.unserved == [] for p in plans)


@pytest.mark.parametrize(
    ("horizons", "message"),
    [
        ([], "non-empty"),
        ([Horizon(48, 1.0, np.ones(3, bool)), Horizon(24, 1.0, np.ones(3, bool))], "increasing"),
        ([Horizon(24, 1.0, np.ones(3, bool)), Horizon(24, 1.0, np.ones(3, bool))], "unique"),
        ([Horizon(24, -1.0, np.ones(3, bool))], "water level"),
        ([Horizon(24, 1.0, np.ones(2, bool))], "eligible"),
        ([Horizon(24, 1.0, np.ones(3, int))], "eligible"),
    ],
)
def test_bad_horizons_are_rejected(horizons, message):
    with pytest.raises(InvalidInputError, match=message):
        plan_evacuation(_scenario(), _sites(), "r1", horizons)


def test_unknown_river_and_bad_sites_are_rejected():
    with pytest.raises(InvalidInputError, match="unknown river_id"):
        plan_evacuation(_scenario(), _sites(), "krueng", _horizons())
    outside = EvacuationSites(
        np.array([1]), np.array([9]), np.array([5.0]), ["school"], np.array([False])
    )
    with pytest.raises(InvalidInputError, match="site nodes"):
        plan_evacuation(_scenario(), outside, "r1", _horizons())


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"nodes": np.array([1, 2])}, "same length"),
        ({"facility_ids": np.array([1, 1, 2])}, "unique"),
        ({"preferred": np.array([0, 1, 0])}, "bool"),
    ],
)
def test_evacuation_sites_validation(changes, message):
    fields = {
        "facility_ids": np.array([1, 2, 3]),
        "nodes": np.array([0, 1, 2]),
        "hand": np.ones(3),
        "types": ["school"] * 3,
        "preferred": np.zeros(3, dtype=bool),
    }
    with pytest.raises(InvalidInputError, match=message):
        EvacuationSites(**(fields | changes))
