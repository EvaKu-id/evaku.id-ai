import time

from fastapi import APIRouter

from evaku_ai.api import schemas
from evaku_ai.api.cache import param_hash
from evaku_ai.api.dependencies import CacheDep, DataVersionDep, ScenarioDep
from evaku_ai.api.responses import DATA_NOT_LOADED
from evaku_ai.core.errors import DataNotLoadedError
from evaku_ai.engine.simulation import SimulationResult, run_simulation

router = APIRouter(tags=["simulation"])

MS_PER_S = 1000


@router.post("/simulation", response_model=schemas.SimulationResponse, responses=DATA_NOT_LOADED)
def simulate(
    request: schemas.SimulationRequest,
    scenario: ScenarioDep,
    cache: CacheDep,
    data_version: DataVersionDep,
) -> schemas.SimulationResponse:
    if scenario is None or cache is None:
        raise DataNotLoadedError("static data is not loaded")
    started = time.perf_counter()
    key = param_hash(data_version, request)
    result = cache.get(key)
    if result is None:
        result = run_simulation(
            scenario, request.river_id, request.water_level_m, request.vehicle, request.hub_ids
        )
        cache.put(key, result)
    elapsed_ms = round((time.perf_counter() - started) * MS_PER_S)
    return _to_response(result, key, data_version, elapsed_ms)


def _to_response(
    result: SimulationResult, key: str, data_version: str, elapsed_ms: int
) -> schemas.SimulationResponse:
    return schemas.SimulationResponse(
        param_hash=key,
        data_version=data_version,
        roads=[
            schemas.RoadChange(edge_id=r.edge_id, status=r.status, depth_cm=r.depth_cm)
            for r in result.roads
        ],
        villages=[
            schemas.VillageResult(
                village_id=v.village_id,
                status=v.status,
                isolated_population=v.isolated_population,
                priority_score=v.priority_score,
            )
            for v in result.villages
        ],
        summary=schemas.Summary(
            isolated_villages=result.summary.isolated_villages,
            isolated_population=result.summary.isolated_population,
            cut_road_km=result.summary.cut_road_km,
        ),
        ms=elapsed_ms,
    )
