import pytest

from evaku_ai.core.errors import ConfigurationError
from evaku_ai.domain.enums import FacilityType, RoadCondition, Vehicle
from evaku_ai.domain.rules import (
    EvacuationRules,
    FacilityRules,
    HandRules,
    IsolationRules,
    PassabilityThresholds,
    RoadNetworkRules,
    RouteRules,
    ScoreWeights,
    passable,
    road_condition,
)


@pytest.mark.parametrize(
    ("depth_cm", "expected"),
    [
        (None, RoadCondition.UNKNOWN),
        (0, RoadCondition.SAFE),
        (10, RoadCondition.SAFE),
        (30, RoadCondition.SAFE),
        (31, RoadCondition.LIMITED),
        (60, RoadCondition.LIMITED),
        (61, RoadCondition.CUT),
    ],
)
def test_road_condition(depth_cm, expected):
    assert road_condition(depth_cm) is expected


def test_road_condition_uses_given_thresholds():
    thresholds = PassabilityThresholds(car_cm=20, truck_cm=40)
    assert road_condition(25, thresholds=thresholds) is RoadCondition.LIMITED
    assert road_condition(41, thresholds=thresholds) is RoadCondition.CUT


@pytest.mark.parametrize(("car_cm", "truck_cm"), [(-1, 60), (70, 60)])
def test_thresholds_reject_invalid_order(car_cm, truck_cm):
    with pytest.raises(ConfigurationError):
        PassabilityThresholds(car_cm=car_cm, truck_cm=truck_cm)


def test_limited_roads_are_truck_only():
    assert passable(RoadCondition.LIMITED, Vehicle.TRUCK)
    assert not passable(RoadCondition.LIMITED, Vehicle.CAR)
    assert not passable(RoadCondition.CUT, Vehicle.TRUCK)


@pytest.mark.parametrize("vehicle", list(Vehicle))
def test_unknown_roads_stay_passable(vehicle):
    assert passable(RoadCondition.UNKNOWN, vehicle)


def test_isolation_rules_defaults():
    rules = IsolationRules()
    assert (rules.fully_share, rules.partial_share, rules.delay_factor) == (0.5, 0.1, 3.0)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"partial_share": 0.0}, "partial_share"),
        ({"partial_share": 0.6}, "partial_share"),
        ({"fully_share": 1.0}, "fully_share"),
        ({"delay_factor": 1.0}, "delay_factor"),
    ],
)
def test_isolation_rules_reject_invalid_values(changes, message):
    with pytest.raises(ConfigurationError, match=message):
        IsolationRules(**changes)


def test_score_weights_sum_to_one():
    weights = ScoreWeights()
    total = (
        weights.isolated_population
        + weights.vulnerable_share
        + weights.no_health_facility
        + weights.isolation_duration
        + weights.access_difficulty
    )
    assert total == pytest.approx(1.0)


@pytest.mark.parametrize(
    "changes",
    [
        {"isolated_population": 0.5},
        {"isolated_population": -0.1, "vulnerable_share": 0.65},
    ],
)
def test_score_weights_reject_invalid_values(changes):
    with pytest.raises(ConfigurationError, match="score weights"):
        ScoreWeights(**changes)


def test_road_network_defaults_and_links():
    rules = RoadNetworkRules()
    assert rules.speed_kmh("primary") == 50.0
    assert rules.speed_kmh("primary_link") == rules.speed_kmh("primary")
    assert rules.speed_kmh("track") == 15.0
    assert rules.speed_kmh("footway") is None
    assert rules.buffer_m == 5000.0


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"speeds_kmh": {}}, "at least one"),
        ({"speeds_kmh": {"primary": 0.0}}, "> 0"),
        ({"speeds_kmh": {"primary": float("nan")}}, "> 0"),
        ({"buffer_m": -1.0}, "buffer_m"),
    ],
)
def test_road_network_rejects_invalid_values(changes, message):
    with pytest.raises(ConfigurationError, match=message):
        RoadNetworkRules(**changes)


def test_facility_rules_defaults():
    rules = FacilityRules()
    assert (rules.overture_min_confidence, rules.dedupe_m, rules.max_snap_m) == (0.5, 50.0, 2000.0)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"overture_min_confidence": 1.5}, "overture_min_confidence"),
        ({"dedupe_m": -1.0}, "dedupe_m"),
        ({"max_snap_m": 0.0}, "max_snap_m"),
    ],
)
def test_facility_rules_reject_invalid_values(changes, message):
    with pytest.raises(ConfigurationError, match=message):
        FacilityRules(**changes)


def test_hand_rules_defaults():
    rules = HandRules()
    assert (rules.cell_size_m, rules.stream_threshold_km2) == (30.0, 1.0)
    assert rules.distance_bands_m == (100.0, 500.0, 2000.0)
    assert (rules.sample_step_m, rules.crossing_buffer_m) == (15.0, 45.0)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"cell_size_m": 0.0}, "> 0"),
        ({"stream_threshold_km2": float("inf")}, "> 0"),
        ({"osm_match_m": -1.0}, "> 0"),
        ({"osm_min_recall": 1.5}, "osm_min_recall"),
        ({"burn_depth_m": -1.0}, "burn_depth_m"),
        ({"sample_step_m": 0.0}, "> 0"),
        ({"crossing_buffer_m": -5.0}, "> 0"),
        ({"village_low_hand_m": ()}, "village_low_hand_m"),
        ({"village_low_hand_m": (1.0, -3.0)}, "village_low_hand_m"),
        ({"burn_depth_m": float("nan")}, "burn_depth_m"),
        ({"distance_bands_m": ()}, "distance_bands_m"),
        ({"distance_bands_m": (500.0, 100.0)}, "distance_bands_m"),
        ({"distance_bands_m": (0.0, 100.0)}, "distance_bands_m"),
    ],
)
def test_hand_rules_reject_invalid_values(changes, message):
    with pytest.raises(ConfigurationError, match=message):
        HandRules(**changes)


def test_evacuation_rules_defaults():
    rules = EvacuationRules()
    assert (rules.safety_margin_m, rules.max_sites, rules.max_distance_m) == (1.0, 3, 10_000.0)
    assert set(rules.site_types) == set(FacilityType)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"safety_margin_m": -0.5}, "safety_margin_m"),
        ({"max_sites": 0}, "max_sites"),
        ({"max_sites": 4}, "max_sites"),
        ({"max_distance_m": 0.0}, "> 0"),
        ({"walk_speed_kmh": float("nan")}, "> 0"),
        ({"site_types": ()}, "at least one"),
        ({"site_types": ("bunker",)}, "unknown site_types"),
    ],
)
def test_evacuation_rules_reject_invalid_values(changes, message):
    with pytest.raises(ConfigurationError, match=message):
        EvacuationRules(**changes)


def test_route_rules_default_boat_speed():
    assert RouteRules().boat_speed_kmh == 10.0


@pytest.mark.parametrize("speed", [0.0, -5.0, float("nan"), float("inf")])
def test_route_rules_reject_invalid_boat_speed(speed):
    with pytest.raises(ConfigurationError, match="boat_speed_kmh"):
        RouteRules(boat_speed_kmh=speed)
