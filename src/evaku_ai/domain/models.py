from dataclasses import dataclass

from evaku_ai.domain.enums import RoadCondition, RouteMode, VillageAccessStatus


@dataclass(frozen=True)
class RoadChange:
    edge_id: int
    status: RoadCondition
    depth_cm: int | None


@dataclass(frozen=True)
class VillageResult:
    village_id: str
    status: VillageAccessStatus
    isolated_population: int
    flooded_population: int
    isolated_under5: int
    isolated_elderly: int
    priority_score: float | None


@dataclass(frozen=True)
class AccessSummary:
    isolated_villages: int
    isolated_population: int
    cut_road_km: float


@dataclass(frozen=True)
class MetricRow:
    model: str
    name: str
    value: float
    note: str | None = None


@dataclass(frozen=True)
class EvacuationSite:
    village_id: str
    rank: int
    facility_id: int
    distance_m: float
    walk_minutes: float
    drive_minutes: float
    leave_before_hour: int | None
    route_edge_ids: tuple[int, ...]


@dataclass(frozen=True)
class EvacuationPlan:
    # unserved lists eligible villages with no safe site within reach at this horizon.
    horizon_hours: int
    sites: list[EvacuationSite]
    unserved: list[str]


@dataclass(frozen=True)
class BoatPoint:
    lon: float
    lat: float
    water_distance_m: float


@dataclass(frozen=True)
class VillageRoute:
    village_id: str
    hub_id: int
    mode: RouteMode
    coordinates: tuple[tuple[float, float], ...]
    edge_ids: tuple[int, ...]
    distance_m: float
    minutes: float
    boat_point: BoatPoint | None
