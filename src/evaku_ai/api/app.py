import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.gzip import GZipMiddleware

from evaku_ai import __version__
from evaku_ai.api.dependencies import require_model_key
from evaku_ai.api.routers import health, recalc, route, simulation
from evaku_ai.core.config import get_settings
from evaku_ai.core.logging import configure_logging

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    app.state.engine_data = None
    log.info("model api started (data %s, model %s)", settings.data_version, settings.model_version)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Evaku Model API",
        version=__version__,
        description="Internal API for simulation and routing. Called only by the Next.js backend.",
        lifespan=lifespan,
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    protected = [Depends(require_model_key)]
    app.include_router(health.router)
    app.include_router(simulation.router, dependencies=protected)
    app.include_router(route.router, dependencies=protected)
    app.include_router(recalc.router, dependencies=protected)
    return app


app = create_app()
