import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request, status
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from evaku_ai import __version__
from evaku_ai.adapters.bundle import MANIFEST_FILE, read_bundle
from evaku_ai.adapters.release import fetch_bundle
from evaku_ai.adapters.storage import r2_storage
from evaku_ai.api.cache import SimulationCache
from evaku_ai.api.dependencies import require_model_key
from evaku_ai.api.routers import health, recalc, route, simulation
from evaku_ai.core.config import Settings, get_settings
from evaku_ai.core.errors import ConfigurationError, DataNotLoadedError, InvalidInputError
from evaku_ai.core.logging import configure_logging
from evaku_ai.engine.synthetic import synthetic_geometry, synthetic_scenario

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.dependency_overrides.get(get_settings, get_settings)()
    configure_logging(settings.log_level, structured=settings.app_env == "production")
    app.state.simulation_cache = SimulationCache(settings.simulation_cache_size)
    if settings.engine_data == "synthetic":
        app.state.engine_data = synthetic_scenario()
        app.state.road_geometry = synthetic_geometry(app.state.engine_data)
        app.state.data_version = "synthetic"
        log.warning("serving generated synthetic data, not a real region")
    else:
        _load_bundle(app, settings)
    log.info(
        "model api started (data %s, model %s)", app.state.data_version, settings.model_version
    )
    yield


def ensure_bundle(settings: Settings) -> Path:
    local = settings.data_dir / "static" / settings.data_version
    if (local / MANIFEST_FILE).is_file() or not settings.r2_configured:
        return local
    started = time.perf_counter()
    try:
        storage = r2_storage(settings, settings.r2_bucket_private)
    except ConfigurationError as exc:
        raise DataNotLoadedError(str(exc)) from exc
    fetch_bundle(storage, settings.data_version, local)
    log.info(
        "downloaded bundle %s from R2 in %.2f s",
        settings.data_version,
        time.perf_counter() - started,
    )
    return local


def _load_bundle(app: FastAPI, settings: Settings) -> None:
    app.state.engine_data = None
    app.state.road_geometry = None
    app.state.data_version = settings.data_version
    started = time.perf_counter()
    try:
        loaded = read_bundle(ensure_bundle(settings), settings.data_version)
    except DataNotLoadedError as exc:
        if settings.app_env == "production":
            raise
        log.warning("static data bundle not loaded (%s); simulation returns 503", exc)
        return
    app.state.engine_data = loaded.data
    app.state.road_geometry = loaded.geometry
    app.state.data_version = loaded.manifest.data_version
    log.info(
        "loaded bundle %s (%d edges, %d villages) in %.2f s",
        loaded.manifest.data_version,
        loaded.data.graph.n_edges,
        len(loaded.data.village_ids),
        time.perf_counter() - started,
    )


async def _data_not_loaded(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


async def _invalid_input(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=status.HTTP_422_UNPROCESSABLE_CONTENT)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Evaku Model API",
        version=__version__,
        description="Internal API for simulation and routing. Called only by the Next.js backend.",
        lifespan=lifespan,
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_exception_handler(DataNotLoadedError, _data_not_loaded)
    app.add_exception_handler(InvalidInputError, _invalid_input)

    protected = [Depends(require_model_key)]
    app.include_router(health.router)
    app.include_router(simulation.router, dependencies=protected)
    app.include_router(route.router, dependencies=protected)
    app.include_router(recalc.router, dependencies=protected)
    return app


app = create_app()
