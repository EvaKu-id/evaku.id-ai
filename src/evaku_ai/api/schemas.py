from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evaku_ai.domain.enums import RoadCondition, Vehicle, VillageAccessStatus


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(_Model):
    status: Literal["ok"] = "ok"
    version: str
    data_version: str
    model_version: str


class SimulationRequest(_Model):
    river_id: str = Field(min_length=1, description="rivers.river_id")
    water_level_m: float = Field(ge=0, le=5, description="Water level above normal, in meters")
    vehicle: Vehicle = Vehicle.CAR
    hub_ids: list[int] | None = Field(
        default=None, description="logistics_hubs.id to route from; default: all active hubs"
    )


class RoadChange(_Model):
    edge_id: int = Field(ge=0)
    status: RoadCondition = Field(description="0 safe, 1 limited, 2 cut, 3 unknown")
    depth_cm: int | None = Field(default=None, ge=0)


class VillageResult(_Model):
    village_id: str
    status: VillageAccessStatus
    isolated_population: int = Field(ge=0)
    priority_score: float | None = Field(default=None, ge=0, le=100)


class Summary(_Model):
    isolated_villages: int = Field(ge=0)
    isolated_population: int = Field(ge=0)
    cut_road_km: float = Field(ge=0)


class SimulationResponse(_Model):
    param_hash: str
    data_version: str
    roads: list[RoadChange] = Field(description="Only segments that are not safe")
    villages: list[VillageResult]
    summary: Summary
    ms: int = Field(ge=0, description="Compute time in milliseconds")


class RouteRequest(_Model):
    village_id: str
    vehicle: Vehicle = Vehicle.CAR
    hub_id: int | None = Field(default=None, description="Default: nearest reachable hub")
    run_id: str | None = Field(default=None, description="Route on a published run")
    param_hash: str | None = Field(default=None, description="Route on a cached simulation")

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "RouteRequest":
        if (self.run_id is None) == (self.param_hash is None):
            raise ValueError("provide exactly one of run_id or param_hash")
        return self


class LineString(_Model):
    type: Literal["LineString"] = "LineString"
    coordinates: list[tuple[float, float]] = Field(description="[lon, lat] pairs, EPSG:4326")


class BoatPoint(_Model):
    lon: float
    lat: float
    water_distance_m: float = Field(ge=0)


class RouteResponse(_Model):
    geojson: LineString
    distance_m: float = Field(ge=0)
    minutes: float = Field(ge=0)
    mode: Literal["land", "land_boat"]
    boat_point: BoatPoint | None = None


class RoadCorrection(_Model):
    edge_id: int = Field(ge=0)
    status: RoadCondition


class RecalcRequest(_Model):
    run_id: str
    corrections: list[RoadCorrection] = Field(min_length=1)
    hub_ids: list[int] | None = None
