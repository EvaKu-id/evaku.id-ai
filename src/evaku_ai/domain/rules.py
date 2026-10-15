import itertools
import math
from collections.abc import Mapping
from dataclasses import astuple, dataclass, field
from types import MappingProxyType

from evaku_ai.core.errors import ConfigurationError
from evaku_ai.domain.enums import FacilityType, RoadCondition, Vehicle


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
class IsolationRules:
    fully_share: float = 0.5
    partial_share: float = 0.1
    delay_factor: float = 3.0

    def __post_init__(self) -> None:
        if not 0 < self.partial_share <= self.fully_share < 1:
            raise ConfigurationError(
                "need 0 < partial_share <= fully_share < 1, "
                f"got partial {self.partial_share}, fully {self.fully_share}"
            )
        if not self.delay_factor > 1:
            raise ConfigurationError(f"delay_factor must be > 1, got {self.delay_factor}")


LINK_SUFFIX = "_link"
ROAD_SPEEDS_KMH: Mapping[str, float] = MappingProxyType(
    {
        "motorway": 80.0,
        "trunk": 60.0,
        "primary": 50.0,
        "secondary": 40.0,
        "tertiary": 35.0,
        "unclassified": 30.0,
        "residential": 25.0,
        "living_street": 15.0,
        "service": 15.0,
        "track": 15.0,
    }
)


@dataclass(frozen=True)
class RoadNetworkRules:
    speeds_kmh: Mapping[str, float] = field(default_factory=lambda: ROAD_SPEEDS_KMH)
    buffer_m: float = 5000.0

    def __post_init__(self) -> None:
        if not self.speeds_kmh:
            raise ConfigurationError("speeds_kmh must list at least one highway type")
        slow = {k: v for k, v in self.speeds_kmh.items() if not (math.isfinite(v) and v > 0)}
        if slow:
            raise ConfigurationError(f"speeds_kmh must be finite and > 0, got {slow}")
        if not (math.isfinite(self.buffer_m) and self.buffer_m >= 0):
            raise ConfigurationError(f"buffer_m must be >= 0, got {self.buffer_m}")

    def speed_kmh(self, highway: str) -> float | None:
        # Link roads (ramps) share the speed of the road class they connect.
        return self.speeds_kmh.get(highway.removesuffix(LINK_SUFFIX))


@dataclass(frozen=True)
class FacilityRules:
    overture_min_confidence: float = 0.5
    dedupe_m: float = 50.0
    max_snap_m: float = 2000.0

    def __post_init__(self) -> None:
        if not 0 <= self.overture_min_confidence <= 1:
            raise ConfigurationError(
                f"overture_min_confidence must be in 0..1, got {self.overture_min_confidence}"
            )
        if not (self.dedupe_m >= 0 and self.max_snap_m > 0):
            raise ConfigurationError(
                f"need dedupe_m >= 0 and max_snap_m > 0, got {self.dedupe_m}, {self.max_snap_m}"
            )


@dataclass(frozen=True)
class HandRules:
    cell_size_m: float = 30.0
    stream_threshold_km2: float = 1.0
    breach_max_dist_m: float = 3000.0
    # Depth carved along mapped rivers so flow follows them across flat floodplains; HAND is
    # measured on the unburned surface, so this only steers routing.
    burn_depth_m: float = 10.0
    # Half a 0.05 degree GloFAS cell: the discharge point can sit anywhere inside its cell.
    pour_point_snap_m: float = 2500.0
    osm_match_m: float = 90.0
    osm_min_recall: float = 0.8
    distance_bands_m: tuple[float, ...] = (100.0, 500.0, 2000.0)
    sample_step_m: float = 15.0
    # Culverts and untagged bridges: samples this close to a road-stream crossing are skipped.
    crossing_buffer_m: float = 45.0
    village_low_hand_m: tuple[float, ...] = (1.0, 3.0)

    def __post_init__(self) -> None:
        sizes = {
            "cell_size_m": self.cell_size_m,
            "stream_threshold_km2": self.stream_threshold_km2,
            "breach_max_dist_m": self.breach_max_dist_m,
            "pour_point_snap_m": self.pour_point_snap_m,
            "osm_match_m": self.osm_match_m,
            "sample_step_m": self.sample_step_m,
            "crossing_buffer_m": self.crossing_buffer_m,
        }
        bad = {k: v for k, v in sizes.items() if not (math.isfinite(v) and v > 0)}
        if bad:
            raise ConfigurationError(f"HAND rules must be finite and > 0, got {bad}")
        if not (math.isfinite(self.burn_depth_m) and self.burn_depth_m >= 0):
            raise ConfigurationError(f"burn_depth_m must be >= 0, got {self.burn_depth_m}")
        if not 0 <= self.osm_min_recall <= 1:
            raise ConfigurationError(f"osm_min_recall must be in 0..1, got {self.osm_min_recall}")
        bands = self.distance_bands_m
        if not bands or bands[0] <= 0 or any(b <= a for a, b in itertools.pairwise(bands)):
            raise ConfigurationError(f"distance_bands_m must be > 0 and increasing, got {bands}")
        low = self.village_low_hand_m
        if not low or any(not (math.isfinite(v) and v > 0) for v in low):
            raise ConfigurationError(f"village_low_hand_m must be finite and > 0, got {low}")


MAX_EVACUATION_SITES = 3


@dataclass(frozen=True)
class EvacuationRules:
    safety_margin_m: float = 1.0
    max_sites: int = MAX_EVACUATION_SITES
    max_distance_m: float = 10_000.0
    walk_speed_kmh: float = 4.0
    site_types: tuple[FacilityType, ...] = tuple(FacilityType)

    def __post_init__(self) -> None:
        if not (math.isfinite(self.safety_margin_m) and self.safety_margin_m >= 0):
            raise ConfigurationError(f"safety_margin_m must be >= 0, got {self.safety_margin_m}")
        if not 1 <= self.max_sites <= MAX_EVACUATION_SITES:
            raise ConfigurationError(
                f"max_sites must be in 1..{MAX_EVACUATION_SITES}, got {self.max_sites}"
            )
        sizes = {"max_distance_m": self.max_distance_m, "walk_speed_kmh": self.walk_speed_kmh}
        bad = {k: v for k, v in sizes.items() if not (math.isfinite(v) and v > 0)}
        if bad:
            raise ConfigurationError(f"evacuation rules must be finite and > 0, got {bad}")
        if not self.site_types:
            raise ConfigurationError("site_types must list at least one facility type")
        unknown = [t for t in self.site_types if t not in set(FacilityType)]
        if unknown:
            raise ConfigurationError(f"unknown site_types {unknown}")


@dataclass(frozen=True)
class ScoreWeights:
    isolated_population: float = 0.35
    vulnerable_share: float = 0.20
    no_health_facility: float = 0.20
    isolation_duration: float = 0.15
    access_difficulty: float = 0.10

    def __post_init__(self) -> None:
        values = astuple(self)
        if any(v < 0 for v in values) or not math.isclose(sum(values), 1.0):
            raise ConfigurationError(f"score weights must be >= 0 and sum to 1, got {values}")


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
    # No depth data means no evidence of flooding; blocking it would isolate villages falsely.
    if condition in (RoadCondition.SAFE, RoadCondition.UNKNOWN):
        return True
    if condition is RoadCondition.LIMITED:
        return vehicle is Vehicle.TRUCK
    return False
