from dataclasses import dataclass

from evaku_ai.domain.enums import Vehicle
from evaku_ai.domain.models import AccessSummary, RoadChange, VillageResult


@dataclass(frozen=True)
class SimulationResult:
    roads: list[RoadChange]
    villages: list[VillageResult]
    summary: AccessSummary


def run_simulation(
    data: object, river_id: str, water_level_m: float, vehicle: Vehicle, hub_ids: list[int] | None
) -> SimulationResult:
    raise NotImplementedError
