import pytest

from evaku_ai.core.errors import ConfigurationError
from evaku_ai.domain.enums import RoadCondition, Vehicle
from evaku_ai.domain.rules import PassabilityThresholds, passable, road_condition


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
    assert not passable(RoadCondition.UNKNOWN, Vehicle.TRUCK)
