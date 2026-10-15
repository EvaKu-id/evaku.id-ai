from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from evaku_ai.adapters.download import fetch
from evaku_ai.adapters.region import load_region
from evaku_ai.core.errors import InvalidInputError
from evaku_ai.domain.region import RegionConfig
from evaku_ai.geo.crs import utm_epsg

log = logging.getLogger(__name__)

WGS84 = "EPSG:4326"
PCODE_PREFIX = "ID"
VILLAGE_ID_DIGITS = 10
DISTRICT_ID_DIGITS = 7
M2_PER_KM2 = 1_000_000
VILLAGE_COLUMNS = ["village_id", "name", "district_id", "district", "regency", "geometry"]
DISTRICT_COLUMNS = ["district_id", "name", "regency", "geometry"]


@dataclass(frozen=True)
class VillageReport:
    villages: int
    districts: int
    area_km2: float
    bounds: tuple[float, float, float, float]
    duplicate_names: list[str]


def read_admin_level(
    path: Path, level: int, bbox: tuple[float, float, float, float]
) -> gpd.GeoDataFrame:
    pattern = re.compile(rf"adm(in)?{level}(?!\d)", re.IGNORECASE)
    names = [str(name) for name, _ in pyogrio.list_layers(path) if pattern.search(str(name))]
    if not names:
        raise InvalidInputError(f"no admin level {level} layer in {path}")
    return gpd.read_file(path, layer=names[0], bbox=bbox)


def select_regency(gdf: gpd.GeoDataFrame, bps_code: str) -> gpd.GeoDataFrame:
    return gdf[gdf["adm2_pcode"] == PCODE_PREFIX + bps_code].copy()


def normalize_villages(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gpd.GeoDataFrame(
        {
            "village_id": _strip_prefix(gdf["adm4_pcode"]),
            "name": gdf["adm4_name"].astype(str).str.strip(),
            "district_id": _strip_prefix(gdf["adm3_pcode"]),
            "district": gdf["adm3_name"].astype(str).str.strip(),
            "regency": gdf["adm2_name"].astype(str).str.strip(),
        },
        geometry=_clean_geometry(gdf),
        crs=WGS84,
    )
    return out[VILLAGE_COLUMNS].reset_index(drop=True)


def normalize_districts(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gpd.GeoDataFrame(
        {
            "district_id": _strip_prefix(gdf["adm3_pcode"]),
            "name": gdf["adm3_name"].astype(str).str.strip(),
            "regency": gdf["adm2_name"].astype(str).str.strip(),
        },
        geometry=_clean_geometry(gdf),
        crs=WGS84,
    )
    return out[DISTRICT_COLUMNS].reset_index(drop=True)


def check_villages(
    villages: gpd.GeoDataFrame, districts: gpd.GeoDataFrame, region: RegionConfig
) -> VillageReport:
    if villages.empty or districts.empty:
        raise InvalidInputError("no villages or districts found for the region")
    ids = villages["village_id"]
    bad_ids = ids[~ids.str.fullmatch(rf"{region.bps_code}\d{{{VILLAGE_ID_DIGITS - 4}}}")]
    if not bad_ids.empty:
        raise InvalidInputError(f"village_id must be {region.bps_code} + 6 digits: {list(bad_ids)}")
    if ids.duplicated().any():
        raise InvalidInputError(f"duplicate village_id: {sorted(set(ids[ids.duplicated()]))}")
    district_ids = villages["district_id"]
    if not district_ids.str.fullmatch(rf"\d{{{DISTRICT_ID_DIGITS}}}").all():
        raise InvalidInputError("district_id must have 7 digits")
    if not all(v.startswith(d) for v, d in zip(ids, district_ids, strict=True)):
        raise InvalidInputError("village_id must start with its district_id")
    missing = set(district_ids) - set(districts["district_id"])
    if missing:
        raise InvalidInputError(f"villages reference unknown districts: {sorted(missing)}")
    blank = (villages["name"].str.len() == 0) | (villages["district"].str.len() == 0)
    if blank.any():
        raise InvalidInputError(f"blank name or district for {list(ids[blank])}")
    for frame in (villages, districts):
        if not (frame.geom_type == "MultiPolygon").all() or not frame.is_valid.all():
            raise InvalidInputError("all geometries must be valid MultiPolygons")
    bounds = tuple(float(v) for v in villages.total_bounds)
    min_lon, min_lat, max_lon, max_lat = region.bbox
    if bounds[0] < min_lon or bounds[1] < min_lat or bounds[2] > max_lon or bounds[3] > max_lat:
        raise InvalidInputError(f"villages extend to {bounds}, outside region bbox {region.bbox}")
    names = Counter(villages["name"])
    area = villages.to_crs(epsg=utm_epsg(*region.center)).area.sum() / M2_PER_KM2
    return VillageReport(
        villages=len(villages),
        districts=len(districts),
        area_km2=round(float(area), 1),
        bounds=(bounds[0], bounds[1], bounds[2], bounds[3]),
        duplicate_names=sorted(name for name, count in names.items() if count > 1),
    )


def build(region_path: Path, out_dir: Path = Path("data/processed")) -> VillageReport:
    region = load_region(region_path)
    source = fetch(region.inputs.village_boundaries)
    villages = normalize_villages(
        select_regency(read_admin_level(source, 4, region.bbox), region.bps_code)
    )
    districts = normalize_districts(
        select_regency(read_admin_level(source, 3, region.bbox), region.bps_code)
    )
    report = check_villages(villages, districts, region)

    out_dir.mkdir(parents=True, exist_ok=True)
    villages.to_parquet(out_dir / "villages.parquet")
    districts.to_parquet(out_dir / "districts.parquet")
    remote = region.inputs.village_boundaries
    meta = {
        "region_id": region.region_id,
        "source": remote.source,
        "license": remote.license,
        "url": str(remote.url),
        "sha256": remote.sha256,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **asdict(report),
    }
    (out_dir / "villages.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info(
        "villages %d, districts %d, area %.1f km2",
        report.villages,
        report.districts,
        report.area_km2,
    )
    return report


def _strip_prefix(codes: pd.Series[str]) -> pd.Series[str]:
    return codes.astype(str).str.strip().str.removeprefix(PCODE_PREFIX)


def _clean_geometry(gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
    geometry = gdf.geometry.to_crs(WGS84) if gdf.crs else gdf.geometry.set_crs(WGS84)
    return geometry.make_valid().apply(_as_multipolygon)


def _as_multipolygon(geom: BaseGeometry) -> MultiPolygon:
    if isinstance(geom, MultiPolygon):
        return geom
    if isinstance(geom, Polygon):
        return MultiPolygon([geom])
    # make_valid can split a polygon into a collection of polygons and stray lines.
    parts = [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon | MultiPolygon)]
    polygons = [p for g in parts for p in (g.geoms if isinstance(g, MultiPolygon) else [g])]
    if not polygons:
        raise InvalidInputError(f"geometry has no polygon area: {geom.geom_type}")
    return MultiPolygon(polygons)
