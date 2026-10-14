import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from evaku_ai.core.config import Settings, get_settings

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
