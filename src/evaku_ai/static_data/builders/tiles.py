import logging
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd

from evaku_ai.adapters import tippecanoe
from evaku_ai.adapters.bundle import read_bundle
from evaku_ai.adapters.download import file_sha256
from evaku_ai.adapters.release import check_release_version
from evaku_ai.adapters.tiles_release import TILES_FILE, TILES_FORMAT, TileFile, TilesManifest
from evaku_ai.core.errors import DataNotLoadedError, InvalidInputError

log = logging.getLogger(__name__)

MAX_ZOOM = 22
MAX_ROADS_BYTES = 50_000_000  # acceptance limit for jalan.pmtiles
BYTES_PER_MB = 1_000_000
WGS84 = "EPSG:4326"
FEATURE_ID = "feature_id"
ROADS_SOURCE = "road_segments.parquet"
VILLAGES_SOURCE = "villages.parquet"


@dataclass(frozen=True)
class TileLayer:
    name: str
    filename: str
    attributes: tuple[str, ...]
    min_zoom: int
    max_zoom: int
    integer_id: str | None = None
    attribution: str = ""

    def __post_init__(self) -> None:
        if not 0 <= self.min_zoom <= self.max_zoom <= MAX_ZOOM:
            raise ValueError(f"invalid zoom range {self.min_zoom}..{self.max_zoom}")
        if self.integer_id is not None and self.integer_id not in self.attributes:
            raise ValueError(f"id attribute {self.integer_id!r} must be in attributes")


ROADS = TileLayer(
    name="ruas",
    filename="jalan.pmtiles",
    attributes=("edge_id", "highway_type", "is_bridge"),
    min_zoom=8,
    max_zoom=14,
    integer_id="edge_id",
    attribution="© OpenStreetMap contributors",
)
VILLAGES = TileLayer(
    name="desa",
    filename="desa.pmtiles",
    attributes=("village_id", "name"),
    min_zoom=7,
    max_zoom=13,
    attribution="Batas desa: BPS via OCHA HDX COD-AB (CC BY-IGO)",
)
SOURCES = {ROADS: ROADS_SOURCE, VILLAGES: VILLAGES_SOURCE}


def tippecanoe_args(
    layer: TileLayer, source: Path, out_dir: Path, binary: Path | str = "tippecanoe"
) -> list[str]:
    args = [
        str(binary),
        "--output",
        str(out_dir / layer.filename),
        "--force",
        "--quiet",
        "--layer",
        layer.name,
        "--minimum-zoom",
        str(layer.min_zoom),
        "--maximum-zoom",
        str(layer.max_zoom),
        "--drop-densest-as-needed",
        f"--description=EvaKu {layer.name}",
        *(f"--include={attribute}" for attribute in layer.attributes),
    ]
    if layer.attribution:
        args.append(f"--attribution={layer.attribution}")
    args.append(str(source))
    return args


def build(
    version: str,
    processed_dir: Path = Path("data/processed"),
    static_dir: Path = Path("data/static"),
    tiles_dir: Path = Path("data/tiles"),
) -> TilesManifest:
    check_release_version(version)
    sources = _check_sources(version, processed_dir, static_dir)
    out = tiles_dir / version
    if (out / TILES_FILE).exists():
        raise InvalidInputError(f"tiles {version} already built at {out}")

    work = tiles_dir / f".{version}.work"
    staging = tiles_dir / f".{version}.part"
    for leftover in (work, staging):
        shutil.rmtree(leftover, ignore_errors=True)
    work.mkdir(parents=True)
    staging.mkdir()
    try:
        binary = tippecanoe.find_binary()
        tool_version = tippecanoe.version(binary)
        files = [
            _build_layer(layer, processed_dir / source, work, staging, binary)
            for layer, source in SOURCES.items()
        ]
        roads = staging / ROADS.filename
        if roads.stat().st_size > MAX_ROADS_BYTES:
            raise InvalidInputError(
                f"{ROADS.filename} is {roads.stat().st_size} bytes, over {MAX_ROADS_BYTES}"
            )
        manifest = TilesManifest(
            format=TILES_FORMAT,
            data_version=version,
            created_at=datetime.now(UTC),
            tippecanoe=tool_version,
            sources=sources,
            files=files,
        )
        part = staging / f"{TILES_FILE}.part"
        part.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        part.replace(staging / TILES_FILE)
        staging.replace(out)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(staging, ignore_errors=True)
    for entry in manifest.files:
        log.info(
            "tiles %s: %s (%s) %d features, %.1f MB",
            version,
            entry.path,
            entry.layer,
            entry.features,
            entry.size / BYTES_PER_MB,
        )
    log.info("built with %s into %s", tool_version, out)
    return manifest


def _check_sources(version: str, processed_dir: Path, static_dir: Path) -> dict[str, str]:
    try:
        bundle = read_bundle(static_dir / version, version)
    except DataNotLoadedError as exc:
        raise InvalidInputError(
            f"build the bundle first (evaku-static build-bundle {version}): {exc}"
        ) from exc
    found = {}
    for name in SOURCES.values():
        path = processed_dir / name
        sha = file_sha256(path) if path.is_file() else None
        if sha is None or sha != bundle.manifest.sources.get(name):
            raise InvalidInputError(
                f"{name} in {processed_dir} changed since bundle {version} was built; "
                "build a new bundle version"
            )
        found[name] = sha
    return found


def _build_layer(
    layer: TileLayer, source: Path, work: Path, staging: Path, binary: Path
) -> TileFile:
    frame = gpd.read_parquet(source, columns=[*layer.attributes, "geometry"])
    if frame.crs is not None and frame.crs.to_string() != WGS84:
        frame = frame.to_crs(WGS84)
    features = work / f"{layer.name}.geojsonl"
    options: dict[str, str] = {}
    if layer.integer_id is not None:
        # GDAL moves ID_FIELD out of the properties, so a copy keeps the id as an attribute too.
        frame[FEATURE_ID] = frame[layer.integer_id]
        options["ID_FIELD"] = FEATURE_ID
    frame.to_file(features, driver="GeoJSONSeq", engine="pyogrio", layer_options=options)
    output = tippecanoe.run(
        tippecanoe_args(layer, features, staging, binary), staging / layer.filename
    )
    return TileFile(
        path=layer.filename,
        layer=layer.name,
        size=output.stat().st_size,
        sha256=file_sha256(output),
        min_zoom=layer.min_zoom,
        max_zoom=layer.max_zoom,
        features=len(frame),
    )
