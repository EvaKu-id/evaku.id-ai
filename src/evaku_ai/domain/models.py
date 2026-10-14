from dataclasses import dataclass

from evaku_ai.domain.enums import RoadCondition, VillageAccessStatus


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
