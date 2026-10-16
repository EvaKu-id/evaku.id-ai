import logging
import shutil
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from evaku_ai.adapters.bundle import read_bundle
from evaku_ai.adapters.download import file_sha256
from evaku_ai.adapters.region import load_region
from evaku_ai.adapters.release import (
    CONTENT_TYPES,
    LATEST_KEY,
    RELEASE_FILE,
    RELEASE_FORMAT,
    LatestPointer,
    ReleaseManifest,
    check_release_version,
    describe_files,
    parse_release,
    read_latest,
    release_prefix,
    verify_release,
    version_key,
)
from evaku_ai.adapters.storage import Storage, r2_storage
from evaku_ai.core.config import get_settings
from evaku_ai.core.errors import DataNotLoadedError, InvalidInputError
from evaku_ai.geo.crs import utm_epsg
from evaku_ai.static_data.builders.village_features import FEATURE_COLUMNS, build_village_features
from evaku_ai.static_data.release_readme import render_readme

log = logging.getLogger(__name__)

GEO_DIR = "geo"
GEO_FILES = (
    "villages.parquet",
    "districts.parquet",
    "road_nodes.parquet",
    "road_segments.parquet",
    "facilities.parquet",
    "logistics_hubs.parquet",
    "node_population.parquet",
    "village_population.parquet",
    "edge_hand.parquet",
    "node_hand.parquet",
    "facility_hand.parquet",
    "village_hand.parquet",
    "streams.parquet",
    "river_basins.parquet",
    "hand.tif",
)
META_FILES = (
    "villages.meta.json",
    "graph.meta.json",
    "facilities.meta.json",
    "population.meta.json",
    "hand.meta.json",
    "hand_objects.meta.json",
)
FEATURES_FILE = "village_static_features.csv"
README_FILE = "README.md"
GIT_TIMEOUT_S = 10.0
LICENSES = (
    "Batas desa dan kecamatan: BPS lewat OCHA HDX COD-AB Indonesia, CC BY-IGO",
    "Penduduk: WorldPop Global2 R2025A (2026), CC BY 4.0, DOI 10.5258/SOTON/WP00841",
    "Jalan, fasilitas OSM, dan sungai referensi: © OpenStreetMap contributors, ODbL",
    "Fasilitas tambahan: Overture Maps Places (CDLA Permissive 2.0, Apache 2.0, CC0)",
    "HAND, sungai, dan DAS: Copernicus DEM GLO-30, © DLR e.V. 2010-2014 dan © Airbus Defence and "
    "Space GmbH 2014-2018, disediakan di bawah COPERNICUS oleh Uni Eropa dan ESA",
)
VERIFY_DIR = Path("data/interim/verify")


def git_state(repo: Path = Path()) -> tuple[str | None, bool]:
    try:
        # Fixed argument lists, no shell; the commit is recorded so data can be traced to code.
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
            check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],  # noqa: S607
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        log.warning("git is not available; the release cannot record its commit")
        return None, True
    if head.returncode != 0 or status.returncode != 0:
        log.warning("not a git checkout; the release cannot record its commit")
        return None, True
    return head.stdout.strip(), bool(status.stdout.strip())


def stage(
    version: str,
    region_path: Path,
    processed_dir: Path = Path("data/processed"),
    static_dir: Path = Path("data/static"),
) -> ReleaseManifest:
    check_release_version(version)
    region = load_region(region_path)
    root = static_dir / version
    try:
        bundle = read_bundle(root, version)
    except DataNotLoadedError as exc:
        raise InvalidInputError(
            f"build the bundle first (evaku-static build-bundle {version}): {exc}"
        ) from exc
    _check_sources(bundle.manifest.sources, processed_dir, version)
    if (root / RELEASE_FILE).exists():
        raise InvalidInputError(f"release {version} is already staged at {root}")
    wanted = [*GEO_FILES, *META_FILES]
    missing = [name for name in wanted if not (processed_dir / name).is_file()]
    if missing:
        raise InvalidInputError(f"missing in {processed_dir}: {missing}")

    geo = root / GEO_DIR
    # A geo/ folder without release.json is leftover from a failed staging, not a release.
    shutil.rmtree(geo, ignore_errors=True)
    geo.mkdir()
    for name in wanted:
        shutil.copy2(processed_dir / name, geo / name)
    features = build_village_features(processed_dir, utm_epsg(*region.center))
    features.to_csv(geo / FEATURES_FILE, index=False)

    (root / README_FILE).unlink(missing_ok=True)
    partial = describe_files(root, exclude={RELEASE_FILE, README_FILE})
    (root / README_FILE).write_text(
        render_readme(version, partial, LICENSES, FEATURE_COLUMNS), encoding="utf-8"
    )

    commit, dirty = git_state()
    if dirty:
        log.warning("working tree has uncommitted changes; the release is recorded as git_dirty")
    manifest = ReleaseManifest(
        format=RELEASE_FORMAT,
        data_version=version,
        region_id=region.region_id,
        created_at=datetime.now(UTC),
        git_commit=commit,
        git_dirty=dirty,
        bundle_format=bundle.manifest.format,
        licenses=list(LICENSES),
        files=describe_files(root, exclude={RELEASE_FILE}),
    )
    part = root / f"{RELEASE_FILE}.part"
    part.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    part.replace(root / RELEASE_FILE)
    log.info("staged release %s: %d files in %s", version, len(manifest.files), root)
    return manifest


def upload(
    version: str,
    storage: Storage,
    static_dir: Path = Path("data/static"),
    resume: bool = False,
) -> None:
    check_release_version(version)
    root = static_dir / version
    release_path = root / RELEASE_FILE
    if not release_path.is_file():
        raise InvalidInputError(f"release {version} is not staged; run the stage step first")
    manifest = parse_release(release_path.read_bytes())
    if manifest.data_version != version:
        raise InvalidInputError(f"{release_path} belongs to {manifest.data_version}")
    for entry in manifest.files:
        local = root / entry.path
        if not local.is_file() or file_sha256(local) != entry.sha256:
            raise InvalidInputError(f"{local} changed since staging; stage a new release version")

    prefix = release_prefix(version)
    remote = {o.key for o in storage.list_objects(prefix)}
    if prefix + RELEASE_FILE in remote:
        raise InvalidInputError(
            f"release {version} already exists in R2; releases are never overwritten"
        )
    if remote and not resume:
        raise InvalidInputError(
            f"found a partial upload of {version} in R2; rerun with --resume to finish it"
        )

    started = time.monotonic()
    sent = 0
    for entry in manifest.files:
        key = prefix + entry.path
        existing = storage.head(key) if resume else None
        if existing is not None:
            if existing.size != entry.size or existing.sha256 != entry.sha256:
                raise InvalidInputError(
                    f"remote {key} differs from the staged file; not overwriting"
                )
            continue
        storage.upload(root / entry.path, key, entry.content_type, entry.sha256)
        sent += entry.size
    storage.upload(
        release_path, prefix + RELEASE_FILE, CONTENT_TYPES[".json"], file_sha256(release_path)
    )
    _advance_latest(storage, version)
    log.info(
        "released %s: %d files (%.1f MB sent) in %.0f s to %s",
        version,
        len(manifest.files) + 1,
        sent / 1e6,
        time.monotonic() - started,
        prefix,
    )


def run(version: str, region: Path, resume: bool = False) -> None:
    settings = get_settings()
    storage = r2_storage(settings, settings.r2_bucket_private)
    if not (Path("data/static") / version / RELEASE_FILE).is_file():
        stage(version, region)
    upload(version, storage, resume=resume)
    log.info(
        "release %s is in bucket %s under %s; latest pointer: %s",
        version,
        settings.r2_bucket_private,
        release_prefix(version),
        LATEST_KEY,
    )


def verify(version: str, dest_root: Path = VERIFY_DIR) -> ReleaseManifest:
    settings = get_settings()
    storage = r2_storage(settings, settings.r2_bucket_private)
    dest = dest_root / version
    shutil.rmtree(dest, ignore_errors=True)
    manifest = verify_release(storage, version, dest)
    total = sum(f.size for f in manifest.files)
    log.info(
        "release %s verified: %d files, %.1f MB, all checksums match",
        version,
        len(manifest.files),
        total / 1e6,
    )
    return manifest


def _check_sources(sources: dict[str, str], processed_dir: Path, version: str) -> None:
    for name, sha in sources.items():
        path = processed_dir / name
        if not path.is_file() or file_sha256(path) != sha:
            raise InvalidInputError(
                f"{name} in {processed_dir} changed since bundle {version} was built; "
                "build a new bundle version"
            )


def _advance_latest(storage: Storage, version: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp)
        current = read_latest(storage, scratch)
        # latest.json only moves forward, so releasing an older version never rolls it back.
        if current is not None and version_key(version) <= version_key(current.data_version):
            log.info("latest stays at %s", current.data_version)
            return
        pointer = LatestPointer(
            data_version=version, released_at=datetime.now(UTC), prefix=release_prefix(version)
        )
        path = scratch / "latest-new.json"
        path.write_text(pointer.model_dump_json(indent=2), encoding="utf-8")
        storage.upload(path, LATEST_KEY, CONTENT_TYPES[".json"], file_sha256(path))
