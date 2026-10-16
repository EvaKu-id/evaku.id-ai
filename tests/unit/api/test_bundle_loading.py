import json
import logging
import secrets
import sys
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from evaku_ai.adapters.bundle import write_bundle
from evaku_ai.adapters.release import ReleaseManifest, describe_files, release_prefix
from evaku_ai.api import app as app_module
from evaku_ai.api.app import create_app
from evaku_ai.core.config import Settings, get_settings
from evaku_ai.core.errors import DataNotLoadedError
from evaku_ai.core.logging import JsonFormatter
from evaku_ai.engine.synthetic import RIVER_ID, synthetic_geometry, synthetic_scenario
from tests.fixtures.evacuation import sample_sites
from tests.fixtures.storage import FakeStorage

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
        synthetic_geometry(synthetic_scenario()),
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


R2_SETTINGS = {
    "r2_endpoint": "https://account.r2.cloudflarestorage.com",
    "r2_access_key_id": secrets.token_urlsafe(8),
    "r2_secret_access_key": secrets.token_urlsafe(16),
}
BODY = {"river_id": RIVER_ID, "water_level_m": 1.0, "vehicle": "truck"}


def _remote_release(tmp_path):
    source = write_bundle(
        synthetic_scenario(),
        sample_sites(),
        synthetic_geometry(synthetic_scenario()),
        tmp_path / "source",
        data_version=VERSION,
        region_id="synthetic",
        sources={},
    )
    files = describe_files(source, exclude=set())
    manifest = ReleaseManifest(
        format=1,
        data_version=VERSION,
        region_id="synthetic",
        created_at=datetime(2026, 10, 16, tzinfo=UTC),
        git_commit=None,
        git_dirty=False,
        bundle_format=2,
        licenses=[],
        files=files,
    )
    storage = FakeStorage()
    prefix = release_prefix(VERSION)
    for entry in files:
        data = (source / entry.path).read_bytes()
        storage.objects[prefix + entry.path] = (data, entry.content_type, entry.sha256)
    storage.objects[prefix + "release.json"] = (
        manifest.model_dump_json().encode(),
        "application/json",
        "0" * 64,
    )
    return storage


def test_missing_local_bundle_is_downloaded_from_r2(tmp_path, monkeypatch):
    storage = _remote_release(tmp_path)
    monkeypatch.setattr(app_module, "r2_storage", lambda settings, bucket: storage)
    data_dir = tmp_path / "data"
    with _client(data_dir, **R2_SETTINGS) as client:
        assert client.get("/health").json()["data_version"] == VERSION
        res = client.post("/simulation", json=BODY, headers={"X-Model-Key": KEY})
    assert res.status_code == 200
    assert (data_dir / "static" / VERSION / "scenario.npz").is_file()
    downloaded = [key.rsplit("/", 1)[-1] for call, key in storage.calls if call == "download"]
    assert sorted(downloaded) == ["manifest.json", "release.json", "scenario.npz"]


def test_local_bundle_is_used_without_touching_r2(tmp_path, monkeypatch):
    storage = FakeStorage()
    monkeypatch.setattr(app_module, "r2_storage", lambda settings, bucket: storage)
    _bundle(tmp_path)
    with _client(tmp_path, **R2_SETTINGS) as client:
        assert client.get("/health").json()["data_version"] == VERSION
    assert storage.calls == []


def test_failed_download_stops_production_but_not_local(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "r2_storage", lambda settings, bucket: FakeStorage())
    with _client(tmp_path / "local", **R2_SETTINGS) as client:
        res = client.post("/simulation", json=BODY, headers={"X-Model-Key": KEY})
    assert res.status_code == 503
    production = _client(tmp_path / "prod", app_env="production", **R2_SETTINGS)
    with pytest.raises(DataNotLoadedError, match="could not download bundle"), production:
        pass


def test_insecure_r2_endpoint_counts_as_a_missing_bundle(tmp_path):
    settings = R2_SETTINGS | {"r2_endpoint": "http://account.r2.cloudflarestorage.com"}
    with _client(tmp_path / "local", **settings) as client:
        res = client.post("/simulation", json=BODY, headers={"X-Model-Key": KEY})
    assert res.status_code == 503
    production = _client(tmp_path / "prod", app_env="production", **settings)
    with pytest.raises(DataNotLoadedError, match="https://"), production:
        pass
