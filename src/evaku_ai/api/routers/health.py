from fastapi import APIRouter

from evaku_ai import __version__
from evaku_ai.api.dependencies import SettingsDep
from evaku_ai.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(settings: SettingsDep) -> HealthResponse:
    return HealthResponse(
        version=__version__,
        data_version=settings.data_version,
        model_version=settings.model_version,
    )
