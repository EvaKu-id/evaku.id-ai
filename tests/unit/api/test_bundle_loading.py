import json
import logging
import secrets
import sys

import pytest
from fastapi.testclient import TestClient

from evaku_ai.adapters.bundle import write_bundle
from evaku_ai.api.app import create_app
from evaku_ai.core.config import Settings, get_settings
from evaku_ai.core.errors import DataNotLoadedError
from evaku_ai.core.logging import JsonFormatter
from evaku_ai.engine.synthetic import RIVER_ID, synthetic_scenario
from tests.fixtures.evacuation import sample_sites

VERSION = "2026-10-16.1"
KEY = secrets.token_urlsafe(16)


def _client(tmp_path, **changes):
    settings = Settings(
        _env_file=None,
        model_api_key=KEY,
        data_dir=tmp_path,
        data_version=VERSION,
        model_version="m1",
        **changes,
    )
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


def _bundle(tmp_path):
    write_bundle(
        synthetic_scenario(),
        sample_sites(),
        tmp_path / "static",
        data_version=VERSION,
        region_id="synthetic",
        sources={},
    )


def test_bundle_is_loaded_and_served(tmp_path):
    _bundle(tmp_path)
    with _client(tmp_path) as client:
        assert client.get("/health").json()["data_version"] == VERSION
        body = {"river_id": RIVER_ID, "water_level_m": 1.0, "vehicle": "truck"}
        res = client.post("/simulation", json=body, headers={"X-Model-Key": KEY})
    assert res.status_code == 200
    assert res.json()["data_version"] == VERSION


def test_missing_bundle_locally_still_starts_with_503(tmp_path):
    with _client(tmp_path) as client:
        body = {"river_id": RIVER_ID, "water_level_m": 1.0, "vehicle": "car"}
        res = client.post("/simulation", json=body, headers={"X-Model-Key": KEY})
    assert res.status_code == 503


def test_missing_bundle_in_production_stops_start(tmp_path):
    client = _client(tmp_path, app_env="production")
    with pytest.raises(DataNotLoadedError, match="no static data bundle"), client:
        pass


def test_json_formatter_writes_one_json_object():
    record = logging.LogRecord("evaku", logging.WARNING, __file__, 1, "loaded %s", ("v1",), None)
    entry = json.loads(JsonFormatter().format(record))
    assert entry["severity"] == "WARNING"
    assert entry["message"] == "loaded v1"
    assert entry["logger"] == "evaku"
    try:
        raise ValueError("boom")
    except ValueError:
        record.exc_info = sys.exc_info()
    assert "ValueError: boom" in json.loads(JsonFormatter().format(record))["exception"]
