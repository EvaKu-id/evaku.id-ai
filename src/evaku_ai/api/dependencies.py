import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status

from evaku_ai.api.cache import SimulationCache
from evaku_ai.core.config import Settings, get_settings
from evaku_ai.engine.simulation import ScenarioData

SettingsDep = Annotated[Settings, Depends(get_settings)]


def require_model_key(
    settings: SettingsDep,
    x_model_key: Annotated[str | None, Header()] = None,
) -> None:
    if settings.model_api_key is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "MODEL_API_KEY is not configured")
    expected = settings.model_api_key.get_secret_value()
    if x_model_key is None or not secrets.compare_digest(x_model_key, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing X-Model-Key")


# These return None instead of raising so request validation (422) still runs first.
def get_scenario(request: Request) -> ScenarioData | None:
    data = getattr(request.app.state, "engine_data", None)
    return data if isinstance(data, ScenarioData) else None


def get_simulation_cache(request: Request) -> SimulationCache | None:
    cache = getattr(request.app.state, "simulation_cache", None)
    return cache if isinstance(cache, SimulationCache) else None


def get_data_version(request: Request) -> str:
    return str(getattr(request.app.state, "data_version", ""))


ScenarioDep = Annotated[ScenarioData | None, Depends(get_scenario)]
CacheDep = Annotated[SimulationCache | None, Depends(get_simulation_cache)]
DataVersionDep = Annotated[str, Depends(get_data_version)]
