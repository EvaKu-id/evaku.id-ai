import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evaku_ai.adapters.download import file_sha256
from evaku_ai.adapters.release import parse_latest
from evaku_ai.adapters.tiles_release import (
    IMMUTABLE,
    MAP_LATEST_KEY,
    NO_CACHE,
    PMTILES_CONTENT_TYPE,
    TILES_FILE,
    TileFile,
    TilesManifest,
    parse_tiles_manifest,
    tiles_prefix,
    upload_tiles,
    verify_tiles,
)
from evaku_ai.core.errors import DownloadError, InvalidInputError
from tests.fixtures.storage import FakeStorage

VERSION = "2026-10-16.3"


def _built(tiles_dir: Path, version: str = VERSION) -> TilesManifest:
    root = tiles_dir / version
    root.mkdir(parents=True)
    files = []
    for name, layer in (("jalan.pmtiles", "ruas"), ("desa.pmtiles", "desa")):
        (root / name).write_bytes(name.encode() * 10)
        files.append(
            TileFile(
                path=name,
                layer=layer,
                size=(root / name).stat().st_size,
                sha256=file_sha256(root / name),
                min_zoom=7,
                max_zoom=14,
                features=3,
            )
        )
    manifest = TilesManifest(
        format=1,
        data_version=version,
        created_at=datetime(2026, 10, 16, tzinfo=UTC),
        tippecanoe="tippecanoe v2.79.0",
        sources={"road_segments.parquet": "a" * 64},
        files=files,
    )
    (root / TILES_FILE).write_text(manifest.model_dump_json(), encoding="utf-8")
    return manifest


def test_upload_sends_tiles_then_manifest_then_latest(tmp_path):
    _built(tmp_path)
    storage = FakeStorage()
    upload_tiles(storage, tmp_path, VERSION)
    prefix = tiles_prefix(VERSION)
    sent = [key for call, key in storage.calls if call == "upload"]
    assert sent == [
        prefix + "jalan.pmtiles",
        prefix + "desa.pmtiles",
        prefix + TILES_FILE,
        MAP_LATEST_KEY,
    ]
    assert storage.objects[prefix + "jalan.pmtiles"][1] == PMTILES_CONTENT_TYPE
    assert storage.cache_control[prefix + "jalan.pmtiles"] == IMMUTABLE
    assert storage.cache_control[prefix + TILES_FILE] == IMMUTABLE
    assert storage.cache_control[MAP_LATEST_KEY] == NO_CACHE
    latest = parse_latest(storage.objects[MAP_LATEST_KEY][0])
    assert (latest.data_version, latest.prefix) == (VERSION, "map/2026-10-16.3/")


def test_latest_only_moves_forward(tmp_path):
    _built(tmp_path)
    _built(tmp_path, "2026-10-15.1")
    storage = FakeStorage()
    upload_tiles(storage, tmp_path, VERSION)
    upload_tiles(storage, tmp_path, "2026-10-15.1")
    assert parse_latest(storage.objects[MAP_LATEST_KEY][0]).data_version == VERSION


def test_existing_version_is_never_overwritten(tmp_path):
    _built(tmp_path)
    storage = FakeStorage()
    upload_tiles(storage, tmp_path, VERSION)
    with pytest.raises(InvalidInputError, match="never overwritten"):
        upload_tiles(storage, tmp_path, VERSION)


def test_tiles_must_be_built_first(tmp_path):
    with pytest.raises(InvalidInputError, match=r"build-tiles 2026-10-16\.3"):
        upload_tiles(FakeStorage(), tmp_path, VERSION)


@pytest.mark.parametrize("change", ["content", "size", "missing"])
def test_files_changed_after_build_are_refused(tmp_path, change):
    _built(tmp_path)
    target = tmp_path / VERSION / "desa.pmtiles"
    if change == "content":
        target.write_bytes(target.read_bytes()[::-1])
    elif change == "size":
        target.write_bytes(b"x")
    else:
        target.unlink()
    storage = FakeStorage()
    with pytest.raises(InvalidInputError, match="changed since it was built"):
        upload_tiles(storage, tmp_path, VERSION)
    assert storage.objects == {}


def test_manifest_of_another_version_is_refused(tmp_path):
    _built(tmp_path)
    (tmp_path / VERSION).rename(tmp_path / "2026-10-17.1")
    with pytest.raises(InvalidInputError, match=r"belongs to 2026-10-16\.3"):
        upload_tiles(FakeStorage(), tmp_path, "2026-10-17.1")


def test_verify_checks_every_file(tmp_path):
    manifest = _built(tmp_path)
    storage = FakeStorage()
    upload_tiles(storage, tmp_path, VERSION)
    assert verify_tiles(storage, VERSION) == manifest
    prefix = tiles_prefix(VERSION)
    data, content_type, _ = storage.objects[prefix + "desa.pmtiles"]
    storage.objects[prefix + "desa.pmtiles"] = (data, content_type, "0" * 64)
    del storage.objects[prefix + "jalan.pmtiles"]
    with pytest.raises(InvalidInputError, match=r"jalan\.pmtiles: missing; desa\.pmtiles: size"):
        verify_tiles(storage, VERSION)


def test_verify_needs_the_manifest_of_that_version(tmp_path):
    with pytest.raises(DownloadError):
        verify_tiles(FakeStorage(), VERSION)
    _built(tmp_path, "2026-10-15.1")
    storage = FakeStorage()
    upload_tiles(storage, tmp_path, "2026-10-15.1")
    storage.objects[tiles_prefix(VERSION) + TILES_FILE] = storage.objects[
        tiles_prefix("2026-10-15.1") + TILES_FILE
    ]
    with pytest.raises(InvalidInputError, match=r"are version 2026-10-15\.1"):
        verify_tiles(storage, VERSION)


def test_parse_rejects_bad_manifests(tmp_path):
    raw = json.loads(_built(tmp_path).model_dump_json())
    with pytest.raises(InvalidInputError, match="format 2"):
        parse_tiles_manifest(json.dumps(raw | {"format": 2}).encode())
    with pytest.raises(InvalidInputError, match="invalid"):
        parse_tiles_manifest(json.dumps(raw | {"surprise": 1}).encode())
    with pytest.raises(InvalidInputError, match="larger than"):
        parse_tiles_manifest(b" " * (1 << 20 + 1))
