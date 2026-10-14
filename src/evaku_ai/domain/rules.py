from dataclasses import dataclass

from evaku_ai.core.errors import ConfigurationError
from evaku_ai.domain.enums import RoadCondition, Vehicle


@dataclass(frozen=True)
class PassabilityThresholds:
    car_cm: int = 30
    truck_cm: int = 60

    def __post_init__(self) -> None:
        if not 0 <= self.car_cm <= self.truck_cm:
            raise ConfigurationError(
                f"need 0 <= car_cm <= truck_cm, got car {self.car_cm}, truck {self.truck_cm}"
            )


@dataclass(frozen=True)
class ScoreWeights:
    isolated_population: float = 0.35
    vulnerable_share: float = 0.20
    no_health_facility: float = 0.20
    isolation_duration: float = 0.15
    access_difficulty: float = 0.10


def road_condition(
    depth_cm: float | None,
    *,
    thresholds: PassabilityThresholds = PassabilityThresholds(),
) -> RoadCondition:
    if depth_cm is None:
        return RoadCondition.UNKNOWN
    if depth_cm <= thresholds.car_cm:
        return RoadCondition.SAFE
    if depth_cm <= thresholds.truck_cm:
        return RoadCondition.LIMITED
    return RoadCondition.CUT


def passable(condition: RoadCondition, vehicle: Vehicle) -> bool:
    if condition is RoadCondition.SAFE:
        return True
    if condition is RoadCondition.LIMITED:
        return vehicle is Vehicle.TRUCK
    return False
