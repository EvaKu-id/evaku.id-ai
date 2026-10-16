import logging
import tempfile
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from evaku_ai.adapters.download import file_sha256
from evaku_ai.adapters.release import (
    CONTENT_TYPES,
    MAX_RELEASE_JSON_BYTES,
    RELEASE_VERSION_PATTERN,
    advance_latest,
    check_release_version,
)
from evaku_ai.adapters.storage import Storage
from evaku_ai.core.errors import InvalidInputError

log = logging.getLogger(__name__)

TILES_FILE = "tiles.json"
TILES_FORMAT = 1
MAP_PREFIX = "map"
MAP_LATEST_KEY = f"{MAP_PREFIX}/latest.json"
IMMUTABLE = "public, max-age=31536000, immutable"
NO_CACHE = "no-cache"
PMTILES_CONTENT_TYPE = CONTENT_TYPES[".pmtiles"]


class TileFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(pattern=r"^[A-Za-z0-9_-]+\.pmtiles$")
    layer: str = Field(min_length=1)
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    min_zoom: int = Field(ge=0, le=22)
    max_zoom: int = Field(ge=0, le=22)
    features: int = Field(ge=0)


class TilesManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format: int
    data_version: str = Field(pattern=RELEASE_VERSION_PATTERN)
    created_at: datetime
    tippecanoe: str
    sources: dict[str, str]
    files: list[TileFile] = Field(min_length=1)


def tiles_prefix(version: str) -> str:
    return f"{MAP_PREFIX}/{version}/"


def parse_tiles_manifest(raw: bytes) -> TilesManifest:
    if len(raw) > MAX_RELEASE_JSON_BYTES:
        raise InvalidInputError(f"{TILES_FILE} is larger than {MAX_RELEASE_JSON_BYTES} bytes")
    try:
        manifest = TilesManifest.model_validate_json(raw)
    except ValidationError as exc:
        raise InvalidInputError(f"{TILES_FILE} is invalid: {exc}") from exc
    if manifest.format != TILES_FORMAT:
        raise InvalidInputError(f"tiles format {manifest.format} is not supported")
    return manifest


def upload_tiles(storage: Storage, tiles_dir: Path, version: str) -> TilesManifest:
    check_release_version(version)
    root = tiles_dir / version
    manifest_path = root / TILES_FILE
    if not manifest_path.is_file():
        raise InvalidInputError(
            f"tiles {version} are not built; run evaku-static build-tiles {version}"
        )
    manifest = parse_tiles_manifest(manifest_path.read_bytes())
    if manifest.data_version != version:
        raise InvalidInputError(f"{manifest_path} belongs to {manifest.data_version}")
    for entry in manifest.files:
        local = root / entry.path
        if not local.is_file() or local.stat().st_size != entry.size:
            raise InvalidInputError(f"{local} changed since it was built; build new tiles")
        if file_sha256(local) != entry.sha256:
            raise InvalidInputError(f"{local} changed since it was built; build new tiles")

    prefix = tiles_prefix(version)
    if storage.head(prefix + TILES_FILE) is not None:
        raise InvalidInputError(
            f"tiles {version} already exist in R2; versions are never overwritten"
        )
    for entry in manifest.files:
        storage.upload(
            root / entry.path, prefix + entry.path, PMTILES_CONTENT_TYPE, entry.sha256, IMMUTABLE
        )
    storage.upload(
        manifest_path,
        prefix + TILES_FILE,
        CONTENT_TYPES[".json"],
        file_sha256(manifest_path),
        IMMUTABLE,
    )
    advance_latest(storage, MAP_LATEST_KEY, version, prefix, NO_CACHE)
    log.info("uploaded tiles %s: %d files to %s", version, len(manifest.files), prefix)
    return manifest


def verify_tiles(storage: Storage, version: str) -> TilesManifest:
    check_release_version(version)
    prefix = tiles_prefix(version)
    with tempfile.TemporaryDirectory() as tmp:
        local = storage.download(
            prefix + TILES_FILE, Path(tmp) / TILES_FILE, MAX_RELEASE_JSON_BYTES
        )
        manifest = parse_tiles_manifest(local.read_bytes())
    if manifest.data_version != version:
        raise InvalidInputError(f"tiles at {prefix} are version {manifest.data_version}")
    bad = []
    for entry in manifest.files:
        remote = storage.head(prefix + entry.path)
        if remote is None:
            bad.append(f"{entry.path}: missing")
        elif remote.size != entry.size or remote.sha256 != entry.sha256:
            bad.append(f"{entry.path}: size or checksum differs")
    if bad:
        raise InvalidInputError(f"tiles {version} failed verification: {'; '.join(bad)}")
    return manifest
