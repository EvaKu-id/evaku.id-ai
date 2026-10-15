import copy
from datetime import date

import pytest
from pydantic import ValidationError

from evaku_ai.core.errors import ConfigurationError
from evaku_ai.domain.enums import FacilityType, HubType
from evaku_ai.domain.region import RegionConfig
from evaku_ai.domain.rules import (
    EvacuationRules,
    FacilityRules,
    HandRules,
    IsolationRules,
    PassabilityThresholds,
    RoadNetworkRules,
    ScoreWeights,
)
from tests.fixtures.region import VALID

WEIGHTS = {
    "isolated_population": 0.4,
    "vulnerable_share": 0.2,
    "no_health_facility": 0.2,
    "isolation_duration": 0.1,
    "access_difficulty": 0.1,
}


def config(**changes):
    data = copy.deepcopy(VALID)
    data.update(changes)
    return RegionConfig.model_validate(data)


def test_valid_region():
    region = config()
    assert region.center == pytest.approx((98.005, 4.21))
    assert region.replay.start_date == date(2026, 10, 10)
    assert region.replay_backup.days_before == 3
    assert region.river("tamiang").glofas_point == (98.125, 4.375)


def test_rules_default_to_domain_values():
    region = config()
    assert region.passability() == PassabilityThresholds()
    assert region.isolation_rules() == IsolationRules()
    assert region.weights() == ScoreWeights()


def test_rule_overrides_are_applied():
    region = config(
        passability_cm={"car_cm": 20, "truck_cm": 50},
        isolation={"fully_share": 0.6, "partial_share": 0.2, "delay_factor": 2.0},
        score_weights=WEIGHTS,
    )
    assert region.passability() == PassabilityThresholds(car_cm=20, truck_cm=50)
    assert region.isolation_rules() == IsolationRules(0.6, 0.2, 2.0)
    assert region.weights() == ScoreWeights(**WEIGHTS)


def test_road_network_defaults_and_override():
    assert config().road_network() == RoadNetworkRules()
    region = config(road_network={"speeds_kmh": {"primary": 40.0, "track": 10.0}, "buffer_m": 2000})
    rules = region.road_network()
    assert (rules.speed_kmh("primary"), rules.speed_kmh("residential"), rules.buffer_m) == (
        40.0,
        None,
        2000.0,
    )


def test_facility_override_and_bpbd_locations():
    assert config().facility_rules() == FacilityRules()
    region = config(
        facilities={"overture_min_confidence": 0.7, "dedupe_m": 30, "max_snap_m": 1000},
        bpbd_locations=[
            {"name": "Gudang BPBD", "lon": 98.05, "lat": 4.28, "hub_type": "bpbd_office"},
            {"name": "SD 1", "lon": 98.06, "lat": 4.28, "facility_type": "school"},
        ],
    )
    assert region.facility_rules().overture_min_confidence == 0.7
    assert region.bpbd_locations[0].hub_type is HubType.BPBD_OFFICE
    assert region.bpbd_locations[1].facility_type is FacilityType.SCHOOL


def test_hand_override_keeps_unset_defaults():
    assert config().hand_rules() == HandRules()
    rules = config(hand={"stream_threshold_km2": 2.5}).hand_rules()
    assert rules.stream_threshold_km2 == 2.5
    assert rules.cell_size_m == HandRules().cell_size_m


def test_evacuation_override():
    assert config().evacuation_rules() == EvacuationRules()
    rules = config(evacuation={"max_distance_m": 5000, "site_types": ["school"]}).evacuation_rules()
    assert rules.max_distance_m == 5000
    assert rules.site_types == (FacilityType.SCHOOL,)
    assert rules.safety_margin_m == 1.0


def test_unknown_river_raises():
    with pytest.raises(ConfigurationError, match="unknown river_id"):
        config().river("krueng")


def _nested(key, **changes):
    value = copy.deepcopy(VALID[key])
    if isinstance(value, list):
        value[0].update(changes)
    else:
        value.update(changes)
    return {key: value}


def _remote(**changes):
    inputs = copy.deepcopy(VALID["inputs"])
    inputs["village_boundaries"].update(changes)
    return {"inputs": inputs}


def _dem_tile(**changes):
    inputs = copy.deepcopy(VALID["inputs"])
    inputs["dem"][0].update(changes)
    return {"inputs": inputs}


def _population(**changes):
    inputs = copy.deepcopy(VALID["inputs"])
    inputs["population"].update(changes)
    return {"inputs": inputs}


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"surprise": 1}, "Extra inputs"),
        (_nested("replay", surprise=1), "Extra inputs"),
        ({"regency_code": "1116"}, "regency_code"),
        ({"region_id": "Aceh Tamiang"}, "region_id"),
        ({"bbox": [98.31, 3.86, 97.70, 4.56]}, "min_lon"),
        ({"bbox": [200.0, 3.86, 201.0, 4.56]}, "out of range"),
        ({"bbox": [97.70, 3.86, 98.31]}, "bbox"),
        ({"dem_bbox": [97.80, 3.60, 98.40, 4.65]}, "dem_bbox must contain bbox"),
        ({"rivers": []}, "rivers"),
        ({"rivers": VALID["rivers"] * 2}, "unique"),
        (_nested("rivers", glofas_point=[99.5, 4.375]), "outside dem_bbox"),
        (_nested("replay", days_before=8), "days_before"),
        (_nested("replay_backup", peak_date=date(2026, 10, 13)), "different event"),
        (_dem_tile(path="/etc/dem.tif"), "under data/"),
        (_dem_tile(path="data/../secrets.tif"), "under data/"),
        (_dem_tile(path="raw/dem.tif"), "under data/"),
        ({"bps_code": "11.16"}, "bps_code"),
        (
            _nested(
                "inputs",
                overture_places={"release": "latest", "path": "data/raw/o.parquet", "license": "x"},
            ),
            "release",
        ),
        ({"bpbd_locations": [{"name": "X", "lon": 98.0, "lat": 4.0}]}, "exactly one"),
        (
            {
                "bpbd_locations": [
                    {
                        "name": "X",
                        "lon": 98.0,
                        "lat": 4.0,
                        "facility_type": "school",
                        "hub_type": "port",
                    }
                ]
            },
            "exactly one",
        ),
        ({"facilities": {"overture_min_confidence": 2, "dedupe_m": 1, "max_snap_m": 1}}, "0..1"),
        ({"road_network": {"speeds_kmh": {"primary": -5.0}, "buffer_m": 0}}, "> 0"),
        (_nested("inputs", osm_extract="data/raw/region.osm.pbf"), "osm_extract"),
        ({"bps_code": "111"}, "bps_code"),
        (_remote(url="http://example.org/admin.gdb.zip"), "https"),
        (_remote(sha256="XYZ"), "sha256"),
        (_remote(path="/srv/admin.gdb.zip"), "under data/"),
        (_population(under5_1km=[]), "under5_1km"),
        (_population(elderly_1km=[]), "elderly_1km"),
        (_population(year=1999), "year"),
        (_population(worldpop="data/raw/pop.tif"), "Extra inputs"),
        (_nested("inputs", dem=[]), "dem"),
        (_nested("inputs", dem="data/raw/dem.tif"), "dem"),
        ({"hand": {"stream_threshold_km2": 0}}, "> 0"),
        ({"hand": {"surprise": 1}}, "Extra inputs"),
        ({"evacuation": {"max_sites": 5}}, "max_sites"),
        ({"evacuation": {"site_types": ["bunker"]}}, "site_types"),
        ({"passability_cm": {"car_cm": 70, "truck_cm": 60}}, "car_cm <= truck_cm"),
        ({"score_weights": {**WEIGHTS, "isolated_population": 0.5}}, "sum to 1"),
        (
            {"isolation": {"fully_share": 1.0, "partial_share": 0.1, "delay_factor": 3.0}},
            "fully_share",
        ),
    ],
)
def test_invalid_region_is_rejected(changes, message):
    with pytest.raises(ValidationError, match=message):
        config(**changes)
