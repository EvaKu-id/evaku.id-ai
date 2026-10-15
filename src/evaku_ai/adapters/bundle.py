import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from evaku_ai.adapters.download import file_sha256
from evaku_ai.core.errors import DataNotLoadedError, InvalidInputError
from evaku_ai.engine.bundle import BUNDLE_FORMAT, bundle_from_arrays, bundle_to_arrays
from evaku_ai.engine.evacuation import EvacuationSites
from evaku_ai.engine.simulation import ScenarioData

VERSION_PATTERN = r"^[0-9A-Za-z._-]{1,64}$"
ARRAYS_FILE = "scenario.npz"
MANIFEST_FILE = "manifest.json"
MAX_ARRAYS_BYTES = 200_000_000
MAX_MANIFEST_BYTES = 64 * 1024


class BundleManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format: int
    data_version: str = Field(pattern=VERSION_PATTERN)
    region_id: str = Field(min_length=1)
    created_at: datetime
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    counts: dict[str, int]
    sources: dict[str, str]


@dataclass(frozen=True)
class LoadedBundle:
    data: ScenarioData
    sites: EvacuationSites
    manifest: BundleManifest


def check_version(version: str) -> str:
    if not re.fullmatch(VERSION_PATTERN, version) or version in {".", ".."}:
        raise InvalidInputError(f"invalid data version {version!r}")
    return version


def write_bundle(
    data: ScenarioData,
    sites: EvacuationSites,
    static_dir: Path,
    *,
    data_version: str,
    region_id: str,
    sources: dict[str, str],
) -> Path:
    out = static_dir / check_version(data_version)
    if out.exists():
        raise InvalidInputError(f"bundle {out} already exists; releases are never overwritten")
    part = static_dir / f".{data_version}.part"
    shutil.rmtree(part, ignore_errors=True)
    part.mkdir(parents=True)
    try:
        np.savez_compressed(part / ARRAYS_FILE, allow_pickle=False, **bundle_to_arrays(data, sites))
        manifest = BundleManifest(
            format=BUNDLE_FORMAT,
            data_version=data_version,
            region_id=region_id,
            created_at=datetime.now(UTC),
            sha256=file_sha256(part / ARRAYS_FILE),
            counts={
                "edges": data.graph.n_edges,
                "nodes": data.graph.n_nodes,
                "villages": len(data.village_ids),
                "hubs": len(data.hub_nodes),
                "health_facilities": int(data.health_nodes.shape[0]),
                "evacuation_sites": int(sites.facility_ids.shape[0]),
            },
            sources=sources,
        )
        (part / MANIFEST_FILE).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        part.replace(out)
    finally:
        shutil.rmtree(part, ignore_errors=True)
    return out


def read_bundle(
    bundle_dir: Path, expected_version: str, max_bytes: int = MAX_ARRAYS_BYTES
) -> LoadedBundle:
    manifest = _manifest(bundle_dir, expected_version)
    arrays_path = bundle_dir / ARRAYS_FILE
    if not arrays_path.is_file():
        raise DataNotLoadedError(f"{arrays_path} is missing")
    if arrays_path.stat().st_size > max_bytes:
        raise DataNotLoadedError(f"{arrays_path} is larger than {max_bytes} bytes")
    if file_sha256(arrays_path) != manifest.sha256:
        raise DataNotLoadedError(f"{arrays_path} does not match the manifest checksum")
    try:
        with np.load(arrays_path, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in stored.files}
        data, sites = bundle_from_arrays(arrays)
    except (ValueError, InvalidInputError) as exc:
        raise DataNotLoadedError(f"bundle {bundle_dir} is invalid: {exc}") from exc
    return LoadedBundle(data, sites, manifest)


def _manifest(bundle_dir: Path, expected_version: str) -> BundleManifest:
    try:
        check_version(expected_version)
    except InvalidInputError as exc:
        raise DataNotLoadedError(str(exc)) from exc
    path = bundle_dir / MANIFEST_FILE
    if not path.is_file():
        raise DataNotLoadedError(f"no static data bundle at {bundle_dir} ({MANIFEST_FILE} missing)")
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        raise DataNotLoadedError(f"{path} is larger than {MAX_MANIFEST_BYTES} bytes")
    try:
        manifest = BundleManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, ValidationError) as exc:
        raise DataNotLoadedError(f"{path} is invalid: {exc}") from exc
    if manifest.format != BUNDLE_FORMAT:
        raise DataNotLoadedError(
            f"bundle format {manifest.format} is not supported (expected {BUNDLE_FORMAT});"
            " rebuild the bundle"
        )
    if manifest.data_version != expected_version:
        found, wanted = manifest.data_version, expected_version
        raise DataNotLoadedError(f"bundle at {bundle_dir} is version {found}, expected {wanted}")
    return manifest
