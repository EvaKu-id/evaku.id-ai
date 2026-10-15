from fastapi import APIRouter

from evaku_ai import __version__
from evaku_ai.api.dependencies import DataVersionDep, SettingsDep
from evaku_ai.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(settings: SettingsDep, data_version: DataVersionDep) -> HealthResponse:
    return HealthResponse(
        version=__version__,
        data_version=data_version or settings.data_version,
        model_version=settings.model_version,
    )
