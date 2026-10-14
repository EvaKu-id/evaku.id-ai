import secrets

import pytest
from fastapi.testclient import TestClient

from evaku_ai.api.app import create_app
from evaku_ai.core.config import Settings, get_settings

TEST_KEY = secrets.token_urlsafe(16)


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, model_api_key=TEST_KEY, data_version="d1", model_version="m1")


@pytest.fixture
def client(settings: Settings) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


@pytest.fixture
def auth() -> dict[str, str]:
    return {"X-Model-Key": TEST_KEY}
