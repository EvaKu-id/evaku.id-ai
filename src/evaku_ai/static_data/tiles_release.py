import logging
from pathlib import Path

from evaku_ai.adapters.storage import r2_storage
from evaku_ai.adapters.tiles_release import (
    MAP_LATEST_KEY,
    TILES_FILE,
    TilesManifest,
    tiles_prefix,
    upload_tiles,
    verify_tiles,
)
from evaku_ai.core.config import get_settings
from evaku_ai.static_data.builders import tiles

log = logging.getLogger(__name__)

TILES_DIR = Path("data/tiles")


def run(version: str, tiles_dir: Path = TILES_DIR) -> TilesManifest:
    settings = get_settings()
    storage = r2_storage(settings, settings.r2_bucket_public)
    if not (tiles_dir / version / TILES_FILE).is_file():
        tiles.build(version, tiles_dir=tiles_dir)
    manifest = upload_tiles(storage, tiles_dir, version)
    log.info(
        "tiles %s are in bucket %s under %s; latest pointer: %s",
        version,
        settings.r2_bucket_public,
        tiles_prefix(version),
        MAP_LATEST_KEY,
    )
    return manifest


def verify(version: str) -> TilesManifest:
    settings = get_settings()
    manifest = verify_tiles(r2_storage(settings, settings.r2_bucket_public), version)
    log.info("tiles %s verified: %d files, all checksums match", version, len(manifest.files))
    return manifest
