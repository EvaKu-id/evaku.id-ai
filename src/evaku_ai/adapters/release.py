import logging
import re
import shutil
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from evaku_ai.adapters.bundle import ARRAYS_FILE, MANIFEST_FILE
from evaku_ai.adapters.download import file_sha256
from evaku_ai.adapters.storage import Storage
from evaku_ai.core.errors import (
    DataNotLoadedError,
    DownloadError,
    InvalidInputError,
    StorageError,
)

log = logging.getLogger(__name__)

STATIC_PREFIX = "data/static"
RELEASE_FILE = "release.json"
LATEST_KEY = f"{STATIC_PREFIX}/latest.json"
RELEASE_FORMAT = 1
RELEASE_VERSION_PATTERN = r"^\d{4}-\d{2}-\d{2}\.\d+$"
PATH_PATTERN = r"^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$"
API_FILES = (MANIFEST_FILE, ARRAYS_FILE)
MAX_RELEASE_JSON_BYTES = 1 << 20
MAX_FILE_BYTES = 500_000_000
CONTENT_TYPES = {
    ".parquet": "application/vnd.apache.parquet",
    ".tif": "image/tiff",
    ".json": "application/json",
    ".csv": "text/csv",
    ".md": "text/markdown",
    ".npz": "application/octet-stream",
    ".pmtiles": "application/vnd.pmtiles",
}


class ReleaseFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(pattern=PATH_PATTERN)
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_type: str

    @field_validator("path")
    @classmethod
    def _no_dot_segments(cls, value: str) -> str:
        if any(part in {".", ".."} for part in value.split("/")):
            raise ValueError(f"path must not contain . or .. segments: {value}")
        return value

    @field_validator("content_type")
    @classmethod
    def _known_type(cls, value: str) -> str:
        if value not in CONTENT_TYPES.values():
            raise ValueError(f"unknown content type {value}")
        return value


class ReleaseManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format: int
    data_version: str = Field(pattern=RELEASE_VERSION_PATTERN)
    region_id: str = Field(min_length=1)
    created_at: datetime
    git_commit: str | None
    git_dirty: bool
    bundle_format: int
    licenses: list[str]
    files: list[ReleaseFile]

    @model_validator(mode="after")
    def _paths_are_unique_and_exclude_the_manifest(self) -> "ReleaseManifest":
        paths = [f.path for f in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("file paths must be unique")
        if RELEASE_FILE in paths:
            raise ValueError(f"{RELEASE_FILE} must not list itself")
        return self


class LatestPointer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    data_version: str = Field(pattern=RELEASE_VERSION_PATTERN)
    released_at: datetime
    prefix: str


def check_release_version(version: str) -> str:
    if not re.fullmatch(RELEASE_VERSION_PATTERN, version):
        raise InvalidInputError(f"release version must look like 2026-10-16.1, got {version!r}")
    try:
        version_key(version)
    except ValueError as exc:
        raise InvalidInputError(f"release version {version!r} has an invalid date") from exc
    return version


def version_key(version: str) -> tuple[date, int]:
    day, number = version.split(".", 1)
    return date.fromisoformat(day), int(number)


def release_prefix(version: str) -> str:
    return f"{STATIC_PREFIX}/{version}/"


def parse_release(raw: bytes) -> ReleaseManifest:
    if len(raw) > MAX_RELEASE_JSON_BYTES:
        raise InvalidInputError(f"{RELEASE_FILE} is larger than {MAX_RELEASE_JSON_BYTES} bytes")
    try:
        manifest = ReleaseManifest.model_validate_json(raw)
    except ValidationError as exc:
        raise InvalidInputError(f"{RELEASE_FILE} is invalid: {exc}") from exc
    if manifest.format != RELEASE_FORMAT:
        raise InvalidInputError(f"release format {manifest.format} is not supported")
    return manifest


def parse_latest(raw: bytes) -> LatestPointer:
    try:
        return LatestPointer.model_validate_json(raw)
    except ValidationError as exc:
        raise InvalidInputError(f"latest.json is invalid: {exc}") from exc


def content_type_for(path: Path) -> str:
    try:
        return CONTENT_TYPES[path.suffix.lower()]
    except KeyError:
        raise InvalidInputError(f"no content type for {path.name}; unexpected file type") from None


def describe_files(root: Path, exclude: set[str]) -> list[ReleaseFile]:
    found = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in exclude:
            continue
        found.append(
            ReleaseFile(
                path=relative,
                size=path.stat().st_size,
                sha256=file_sha256(path),
                content_type=content_type_for(path),
            )
        )
    return found


def read_latest(storage: Storage, scratch: Path, key: str = LATEST_KEY) -> LatestPointer | None:
    if storage.head(key) is None:
        return None
    local = storage.download(key, scratch / "latest.json", MAX_RELEASE_JSON_BYTES)
    return parse_latest(local.read_bytes())


def advance_latest(
    storage: Storage, key: str, version: str, prefix: str, cache_control: str | None = None
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp)
        current = read_latest(storage, scratch, key)
        # latest.json only moves forward, so releasing an older version never rolls it back.
        if current is not None and version_key(version) <= version_key(current.data_version):
            log.info("%s stays at %s", key, current.data_version)
            return
        pointer = LatestPointer(data_version=version, released_at=datetime.now(UTC), prefix=prefix)
        path = scratch / "latest-new.json"
        path.write_text(pointer.model_dump_json(indent=2), encoding="utf-8")
        storage.upload(path, key, CONTENT_TYPES[".json"], file_sha256(path), cache_control)


def fetch_bundle(storage: Storage, version: str, dest: Path) -> Path:
    try:
        check_release_version(version)
    except InvalidInputError as exc:
        raise DataNotLoadedError(str(exc)) from exc
    if dest.exists():
        if dest.is_dir() and not any(dest.iterdir()):
            dest.rmdir()
        else:
            raise DataNotLoadedError(f"{dest} exists but holds no valid bundle; remove it first")
    staging = dest.with_name(f".{dest.name}.part")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    prefix = release_prefix(version)
    try:
        raw = storage.download(
            prefix + RELEASE_FILE, staging / RELEASE_FILE, MAX_RELEASE_JSON_BYTES
        )
        manifest = parse_release(raw.read_bytes())
        if manifest.data_version != version:
            raise DataNotLoadedError(
                f"release at {prefix} is version {manifest.data_version}, expected {version}"
            )
        entries = {f.path: f for f in manifest.files}
        for name in API_FILES:
            entry = entries.get(name)
            if entry is None:
                raise DataNotLoadedError(f"release {version} does not list {name}")
            _download_checked(storage, prefix, entry, staging)
        staging.replace(dest)
    except (DownloadError, StorageError, InvalidInputError) as exc:
        raise DataNotLoadedError(f"could not download bundle {version} from R2: {exc}") from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return dest


def verify_release(storage: Storage, version: str, dest: Path) -> ReleaseManifest:
    check_release_version(version)
    prefix = release_prefix(version)
    dest.mkdir(parents=True, exist_ok=True)
    raw = storage.download(prefix + RELEASE_FILE, dest / RELEASE_FILE, MAX_RELEASE_JSON_BYTES)
    manifest = parse_release(raw.read_bytes())
    if manifest.data_version != version:
        raise InvalidInputError(f"release at {prefix} is version {manifest.data_version}")
    expected = {prefix + f.path for f in manifest.files} | {prefix + RELEASE_FILE}
    extra = sorted(o.key for o in storage.list_objects(prefix) if o.key not in expected)
    if extra:
        raise InvalidInputError(f"unexpected objects in release {version}: {extra[:5]}")
    bad = []
    for entry in manifest.files:
        try:
            _download_checked(storage, prefix, entry, dest)
        except (DataNotLoadedError, DownloadError) as exc:
            bad.append(f"{entry.path}: {exc}")
    if bad:
        raise InvalidInputError(f"release {version} failed verification: {'; '.join(bad)}")
    return manifest


def _download_checked(storage: Storage, prefix: str, entry: ReleaseFile, root: Path) -> Path:
    if entry.size > MAX_FILE_BYTES:
        raise DataNotLoadedError(f"{entry.path} is {entry.size} bytes, over {MAX_FILE_BYTES}")
    local = storage.download(prefix + entry.path, root / entry.path, entry.size)
    if local.stat().st_size != entry.size:
        raise DataNotLoadedError(f"{entry.path} has the wrong size")
    if file_sha256(local) != entry.sha256:
        raise DataNotLoadedError(f"{entry.path} does not match its checksum")
    return local
