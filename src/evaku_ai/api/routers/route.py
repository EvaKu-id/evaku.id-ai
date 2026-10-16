from fastapi import APIRouter, HTTPException, status

from evaku_ai.api import schemas
from evaku_ai.api.cache import SimulationCache, cached_simulation
from evaku_ai.api.dependencies import CacheDep, DataVersionDep, GeometryDep, ScenarioDep
from evaku_ai.api.responses import ROUTE_ERRORS
from evaku_ai.core.errors import DataNotLoadedError, InvalidInputError
from evaku_ai.domain.models import VillageRoute
from evaku_ai.engine.routing import plan_routes
from evaku_ai.engine.simulation import ScenarioData, SimulationResult

router = APIRouter(tags=["route"])


@router.post("/route", response_model=schemas.RouteResponse, responses=ROUTE_ERRORS)
def route(
    request: schemas.RouteRequest,
    scenario: ScenarioDep,
    geometry: GeometryDep,
    cache: CacheDep,
    data_version: DataVersionDep,
) -> schemas.RouteResponse:
    if request.run_id is not None:
        raise HTTPException(
            status.HTTP_501_NOT_IMPLEMENTED,
            "Routing on a published run is not available yet; use param_hash",
        )
    if scenario is None or geometry is None or cache is None:
        raise DataNotLoadedError("static data is not loaded")
    if request.village_id not in scenario.village_ids:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown village_id {request.village_id!r}")
    village = scenario.village_ids.index(request.village_id)
    result = _simulation(request, scenario, cache, data_version)
    hubs = _hubs(request.hub_id, scenario, result)
    [found] = plan_routes(scenario, geometry, result.edge_status, request.vehicle, [village], hubs)
    return _to_response(found)


def _simulation(
    request: schemas.RouteRequest,
    scenario: ScenarioData,
    cache: SimulationCache,
    data_version: str,
) -> SimulationResult:
    key = request.param_hash or ""
    result = cache.get(key)
    if result is not None:
        return result
    if request.simulation is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"simulation {key} is not cached; send `simulation` with the request",
        )
    rebuilt, result = cached_simulation(cache, scenario, data_version, request.simulation)
    if rebuilt != key:
        raise InvalidInputError("simulation does not match param_hash")
    return result


def _hubs(hub_id: int | None, scenario: ScenarioData, result: SimulationResult) -> dict[int, int]:
    if hub_id is None:
        return dict(result.hub_nodes)
    if hub_id not in scenario.hub_nodes:
        raise InvalidInputError(f"unknown hub_id {hub_id}")
    return {hub_id: scenario.hub_nodes[hub_id]}


def _to_response(found: VillageRoute) -> schemas.RouteResponse:
    boat = found.boat_point
    return schemas.RouteResponse(
        geojson=schemas.LineString(coordinates=list(found.coordinates)),
        distance_m=found.distance_m,
        minutes=found.minutes,
        mode=found.mode,
        boat_point=None
        if boat is None
        else schemas.BoatPoint(lon=boat.lon, lat=boat.lat, water_distance_m=boat.water_distance_m),
    )
