from datetime import date, timedelta
from pathlib import Path
from typing import Annotated, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, HttpUrl, model_validator

from evaku_ai.core.errors import ConfigurationError
from evaku_ai.domain.enums import FacilityType, HubType
from evaku_ai.domain.rules import (
    EvacuationRules,
    FacilityRules,
    HandRules,
    IsolationRules,
    PassabilityThresholds,
    RoadNetworkRules,
    RouteRules,
    ScoreWeights,
)

SLUG = r"^[a-z0-9-]+$"
MAX_REPLAY_DAYS_BEFORE = 7


def _check_point(point: tuple[float, float]) -> tuple[float, float]:
    lon, lat = point
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        raise ValueError(f"[lon, lat] out of range: {point}")
    return point


def _check_bbox(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    min_lon, min_lat, max_lon, max_lat = bbox
    _check_point((min_lon, min_lat))
    _check_point((max_lon, max_lat))
    if not (min_lon < max_lon and min_lat < max_lat):
        raise ValueError(f"bbox must be [min_lon, min_lat, max_lon, max_lat], got {bbox}")
    return bbox


def _check_data_path(path: Path) -> Path:
    if path.is_absolute() or ".." in path.parts or path.parts[:1] != ("data",):
        raise ValueError(f"input path must be relative and under data/, got {path}")
    return path


def _check_https(url: HttpUrl) -> HttpUrl:
    if url.scheme != "https":
        raise ValueError(f"download url must use https, got {url.scheme}")
    return url


GeoPoint = Annotated[tuple[float, float], AfterValidator(_check_point)]
Bbox = Annotated[tuple[float, float, float, float], AfterValidator(_check_bbox)]
DataPath = Annotated[Path, AfterValidator(_check_data_path)]
HttpsUrl = Annotated[HttpUrl, AfterValidator(_check_https)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class River(_Model):
    river_id: str = Field(pattern=SLUG)
    name: str = Field(min_length=1)
    glofas_point: GeoPoint


class ReplayEvent(_Model):
    name: str = Field(min_length=1)
    peak_date: date
    days_before: int = Field(default=3, ge=0, le=MAX_REPLAY_DAYS_BEFORE)

    @property
    def start_date(self) -> date:
        return self.peak_date - timedelta(days=self.days_before)


class RemoteFile(_Model):
    path: DataPath
    url: HttpsUrl
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: str = Field(min_length=1)
    license: str = Field(min_length=1)


class OvertureSource(_Model):
    release: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}\.\d+$")
    path: DataPath
    license: str = Field(min_length=1)


class ManualLocation(_Model):
    name: str = Field(min_length=1)
    lon: float = Field(ge=-180, le=180)
    lat: float = Field(ge=-90, le=90)
    facility_type: FacilityType | None = None
    hub_type: HubType | None = None

    @model_validator(mode="after")
    def _one_type(self) -> Self:
        if (self.facility_type is None) == (self.hub_type is None):
            raise ValueError(f"{self.name}: set exactly one of facility_type or hub_type")
        return self


class PopulationSource(_Model):
    year: int = Field(ge=2000, le=2100)
    total_100m: RemoteFile
    total_1km: RemoteFile
    under5_1km: list[RemoteFile] = Field(min_length=1)
    elderly_1km: list[RemoteFile] = Field(min_length=1)


class Inputs(_Model):
    osm_extract: RemoteFile
    population: PopulationSource
    dem: list[RemoteFile] = Field(min_length=1)
    village_boundaries: RemoteFile
    overture_places: OvertureSource


class PassabilityOverride(_Model):
    car_cm: int = Field(ge=0)
    truck_cm: int = Field(ge=0)


class IsolationOverride(_Model):
    fully_share: float
    partial_share: float
    delay_factor: float


class RoadNetworkOverride(_Model):
    speeds_kmh: dict[str, float]
    buffer_m: float


class FacilityOverride(_Model):
    overture_min_confidence: float
    dedupe_m: float
    max_snap_m: float


class HandOverride(_Model):
    cell_size_m: float | None = None
    stream_threshold_km2: float | None = None
    breach_max_dist_m: float | None = None
    burn_depth_m: float | None = None
    pour_point_snap_m: float | None = None
    osm_match_m: float | None = None
    osm_min_recall: float | None = None
    distance_bands_m: tuple[float, ...] | None = None
    sample_step_m: float | None = None
    crossing_buffer_m: float | None = None
    village_low_hand_m: tuple[float, ...] | None = None


class EvacuationOverride(_Model):
    safety_margin_m: float | None = None
    max_sites: int | None = None
    max_distance_m: float | None = None
    walk_speed_kmh: float | None = None
    site_types: tuple[FacilityType, ...] | None = None


class RouteOverride(_Model):
    boat_speed_kmh: float | None = None


class ScoreWeightsOverride(_Model):
    isolated_population: float
    vulnerable_share: float
    no_health_facility: float
    isolation_duration: float
    access_difficulty: float


class RegionConfig(_Model):
    region_id: str = Field(pattern=SLUG)
    name: str = Field(min_length=1)
    province: str = Field(min_length=1)
    regency_code: str = Field(pattern=r"^\d{2}\.\d{2}$")
    bps_code: str = Field(pattern=r"^\d{4}$")
    bbox: Bbox
    dem_bbox: Bbox
    rivers: list[River] = Field(min_length=1)
    replay: ReplayEvent
    replay_backup: ReplayEvent | None = None
    inputs: Inputs
    passability_cm: PassabilityOverride | None = None
    isolation: IsolationOverride | None = None
    score_weights: ScoreWeightsOverride | None = None
    road_network_override: RoadNetworkOverride | None = Field(default=None, alias="road_network")
    facility_override: FacilityOverride | None = Field(default=None, alias="facilities")
    hand_override: HandOverride | None = Field(default=None, alias="hand")
    evacuation_override: EvacuationOverride | None = Field(default=None, alias="evacuation")
    route_override: RouteOverride | None = Field(default=None, alias="routing")
    bpbd_locations: list[ManualLocation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if not _contains(self.dem_bbox, self.bbox):
            raise ValueError("dem_bbox must contain bbox")
        ids = [river.river_id for river in self.rivers]
        if len(ids) != len(set(ids)):
            raise ValueError(f"river_id values must be unique, got {ids}")
        for river in self.rivers:
            if not _contains_point(self.dem_bbox, river.glofas_point):
                raise ValueError(f"glofas_point of {river.river_id} is outside dem_bbox")
        if self.replay_backup and self.replay_backup.peak_date == self.replay.peak_date:
            raise ValueError("replay_backup must be a different event from replay")
        # Rule objects validate themselves; surface their errors as validation errors.
        try:
            self.passability()
            self.isolation_rules()
            self.weights()
            self.road_network()
            self.facility_rules()
            self.hand_rules()
            self.evacuation_rules()
            self.route_rules()
        except ConfigurationError as exc:
            raise ValueError(str(exc)) from exc
        return self

    @property
    def center(self) -> tuple[float, float]:
        min_lon, min_lat, max_lon, max_lat = self.bbox
        return (min_lon + max_lon) / 2, (min_lat + max_lat) / 2

    def river(self, river_id: str) -> River:
        for river in self.rivers:
            if river.river_id == river_id:
                return river
        raise ConfigurationError(f"unknown river_id {river_id!r} in region {self.region_id}")

    def passability(self) -> PassabilityThresholds:
        if self.passability_cm is None:
            return PassabilityThresholds()
        return PassabilityThresholds(**self.passability_cm.model_dump())

    def isolation_rules(self) -> IsolationRules:
        if self.isolation is None:
            return IsolationRules()
        return IsolationRules(**self.isolation.model_dump())

    def road_network(self) -> RoadNetworkRules:
        if self.road_network_override is None:
            return RoadNetworkRules()
        return RoadNetworkRules(**self.road_network_override.model_dump())

    def facility_rules(self) -> FacilityRules:
        if self.facility_override is None:
            return FacilityRules()
        return FacilityRules(**self.facility_override.model_dump())

    def hand_rules(self) -> HandRules:
        if self.hand_override is None:
            return HandRules()
        return HandRules(**self.hand_override.model_dump(exclude_none=True))

    def evacuation_rules(self) -> EvacuationRules:
        if self.evacuation_override is None:
            return EvacuationRules()
        return EvacuationRules(**self.evacuation_override.model_dump(exclude_none=True))

    def route_rules(self) -> RouteRules:
        if self.route_override is None:
            return RouteRules()
        return RouteRules(**self.route_override.model_dump(exclude_none=True))

    def weights(self) -> ScoreWeights:
        if self.score_weights is None:
            return ScoreWeights()
        return ScoreWeights(**self.score_weights.model_dump())


def _contains(
    outer: tuple[float, float, float, float], inner: tuple[float, float, float, float]
) -> bool:
    lower_ok = outer[0] <= inner[0] and outer[1] <= inner[1]
    return lower_ok and inner[2] <= outer[2] and inner[3] <= outer[3]


def _contains_point(bbox: tuple[float, float, float, float], point: tuple[float, float]) -> bool:
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]
