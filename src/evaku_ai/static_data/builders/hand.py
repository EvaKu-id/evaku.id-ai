from __future__ import annotations

import itertools
import json
import logging
import math
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.shutil
from rasterio.features import geometry_mask, rasterize, shapes
from rasterio.merge import merge
from rasterio.transform import Affine, rowcol
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from evaku_ai.adapters import whitebox
from evaku_ai.adapters.download import fetch
from evaku_ai.adapters.region import load_region
from evaku_ai.core.errors import ConfigurationError, InvalidInputError
from evaku_ai.domain.arrays import BoolArray, FloatArray, IntArray
from evaku_ai.domain.region import RegionConfig, River
from evaku_ai.domain.rules import HandRules
from evaku_ai.geo.crs import utm_epsg
from evaku_ai.geo.hand import (
    M2_PER_KM2,
    burn,
    drainage_target,
    fill_unreached,
    median_by_distance,
    restore_burn,
    sea_mask,
    snap_pour_point,
    stream_agreement,
    threshold_cells,
    touches_edge,
    trace_downstream,
)

log = logging.getLogger(__name__)

WGS84 = "EPSG:4326"
NODATA = -9999.0
TOLERANCE_M = 1e-3
OSM_WATERWAYS = ("river", "stream")
TOOLS_DIR = Path("data/tools/bin")


@dataclass(frozen=True)
class Raster:
    values: FloatArray
    transform: Affine
    epsg: int


@dataclass(frozen=True)
class Flow:
    pointer_path: Path
    pointer: IntArray
    accumulation: FloatArray
    land: BoolArray
    transform: Affine
    epsg: int


@dataclass(frozen=True)
class HandReport:
    cell_size_m: float
    stream_threshold_km2: float
    stream_threshold_cells: int
    basin_area_km2: dict[str, float]
    regency_share_in_basin: dict[str, float]
    outlets: dict[str, tuple[float, float]]
    median_by_band_m: dict[str, float]
    hand_p5: float
    hand_p50: float
    hand_p95: float
    sea_drained_cells: int
    burned_river_km: float
    burn_clipped_cells: int
    osm_river_recall: float
    osm_precision: float


def prepare_dem(
    tiles: list[Path],
    dem_bbox: tuple[float, float, float, float],
    epsg: int,
    cell_m: float,
    out: Path,
) -> Path:
    sources = [rasterio.open(path) for path in tiles]
    try:
        covered = unary_union([box(*src.bounds) for src in sources])
        if not covered.buffer(1e-9).contains(box(*dem_bbox)):
            raise ConfigurationError(f"DEM tiles do not cover dem_bbox {dem_bbox}")
        mosaic, src_transform = merge(sources, bounds=dem_bbox, nodata=NODATA)
        src_crs = sources[0].crs
    finally:
        for src in sources:
            src.close()
    height, width = mosaic.shape[1:]
    dst_crs = f"EPSG:{epsg}"
    transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, width, height, *dem_bbox, resolution=cell_m
    )
    dem = np.full((dst_height, dst_width), NODATA, dtype=np.float32)
    reproject(
        mosaic[0],
        dem,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=NODATA,
        dst_transform=transform,
        dst_crs=dst_crs,
        dst_nodata=NODATA,
        resampling=Resampling.bilinear,
    )
    _write(out, dem, transform, epsg)
    return out


def condition_dem(
    dem_path: Path, rivers: gpd.GeoDataFrame, depth_m: float, out: Path
) -> tuple[Path, BoolArray]:
    dem = _read(dem_path)
    mask = np.zeros(dem.values.shape, dtype=bool)
    if not rivers.empty:
        # all_touched keeps diagonal river lines 8-connected on the grid.
        mask = rasterize(
            list(rivers.geometry),
            out_shape=mask.shape,
            transform=dem.transform,
            all_touched=True,
        ).astype(bool)
    land = np.isfinite(dem.values) & ~sea_mask(dem.values, NODATA, keep=mask)
    mask &= land
    burned = burn(dem.values, mask, depth_m)
    _write(out, np.where(land, burned, NODATA), dem.transform, dem.epsg)
    return out, mask


def add_mapped_rivers(streams_path: Path, river_cells: BoolArray, out: Path) -> Path:
    # D8 sends a delta fork's flow down one branch only; mapped rivers are drainage regardless.
    streams = _read(streams_path)
    land = np.isfinite(streams.values)
    merged = np.where(land, ((streams.values > 0) | river_cells) * 1.0, NODATA)
    _write(out, merged, streams.transform, streams.epsg)
    return out


def locate_outlet(
    pointer: IntArray,
    accumulation: FloatArray,
    land: BoolArray,
    transform: Affine,
    xy: tuple[float, float],
    snap_cells: int,
) -> tuple[int, int]:
    row, col = rowcol(transform, *xy)
    snapped = snap_pour_point(accumulation, int(row), int(col), snap_cells)
    return trace_downstream(pointer, land, *snapped)


def basin_polygon(mask: BoolArray, transform: Affine, epsg: int) -> gpd.GeoSeries:
    parts = [
        shape(geom) for geom, _ in shapes(mask.astype(np.uint8), mask=mask, transform=transform)
    ]
    return gpd.GeoSeries([unary_union(parts)], crs=f"EPSG:{epsg}").to_crs(WGS84)


def osm_waterways(path: Path, bbox: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    lines = gpd.read_file(path, layer="lines", bbox=bbox, columns=["osm_id", "waterway"])
    lines = lines.set_crs(WGS84, allow_override=True)
    return lines[lines["waterway"].isin(OSM_WATERWAYS)].reset_index(drop=True)


def check_hand(
    hand: FloatArray,
    streams: BoolArray,
    land: BoolArray,
    regency: BoolArray,
    rules: HandRules,
) -> tuple[FloatArray, tuple[float, float, float]]:
    on_streams = hand[streams & land]
    if on_streams.size == 0:
        raise InvalidInputError("no stream cells on land; lower stream_threshold_km2")
    if np.nanmax(np.abs(on_streams)) > TOLERANCE_M:
        raise InvalidInputError("HAND on stream cells must be 0")
    if np.nanmin(hand[land]) < -TOLERANCE_M:
        raise InvalidInputError(f"negative HAND on land: {np.nanmin(hand[land]):.3f} m")
    if (regency & ~np.isfinite(hand)).any():
        raise InvalidInputError("HAND has no data on land inside the regency")
    medians = median_by_distance(hand, streams, land, rules.distance_bands_m, rules.cell_size_m)
    if not (np.isfinite(medians).all() and (np.diff(medians) > 0).all()):
        raise InvalidInputError(f"median HAND must rise away from streams, got {medians.tolist()}")
    inside = hand[regency & np.isfinite(hand)]
    p5, p50, p95 = (float(v) for v in np.percentile(inside, [5, 50, 95]))
    return medians, (p5, p50, p95)


def build(region_path: Path, out_dir: Path = Path("data/processed")) -> HandReport:
    started = time.monotonic()
    region = load_region(region_path)
    rules = region.hand_rules()
    epsg = utm_epsg(*region.center)
    districts = _districts(out_dir)
    tiles = [fetch(tile) for tile in region.inputs.dem]
    tool = whitebox.WHITEBOX_TOOLS
    binary = whitebox.install(fetch(tool), TOOLS_DIR / tool.sha256[:12])
    work_root = Path("data/interim")
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work_root) as tmp:
        work = Path(tmp).resolve()
        report, hand, streams, basins = _hydrology(
            region, rules, epsg, districts, tiles, binary, work
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_cog(hand, out_dir / "hand.tif", work)
    streams.to_parquet(out_dir / "streams.parquet")
    basins.to_parquet(out_dir / "river_basins.parquet")
    meta = {
        "region_id": region.region_id,
        "dem": [{"file": t.path.name, "sha256": t.sha256} for t in region.inputs.dem],
        "dem_source": region.inputs.dem[0].source,
        "dem_license": region.inputs.dem[0].license,
        "dem_bbox": region.dem_bbox,
        "crs": f"EPSG:{epsg}",
        "nodata": NODATA,
        "whitebox": {"version": whitebox.version(binary), "sha256": whitebox.WHITEBOX_TOOLS.sha256},
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seconds": round(time.monotonic() - started, 1),
        **asdict(report),
    }
    (out_dir / "hand.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info(
        "HAND p50 %.1f m, basins %s km2, OSM river recall %.2f",
        report.hand_p50,
        report.basin_area_km2,
        report.osm_river_recall,
    )
    return report


def _hydrology(
    region: RegionConfig,
    rules: HandRules,
    epsg: int,
    districts: gpd.GeoDataFrame,
    tiles: list[Path],
    binary: Path,
    work: Path,
) -> tuple[HandReport, Path, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    cell = rules.cell_size_m
    threshold = threshold_cells(rules.stream_threshold_km2, cell)
    waterways = osm_waterways(region.inputs.osm_extract.path, region.dem_bbox).to_crs(epsg=epsg)
    rivers = waterways.loc[waterways["waterway"] == "river"]
    dem_path = prepare_dem(tiles, region.dem_bbox, epsg, cell, work / "dem.tif")
    burned_path, river_cells = condition_dem(
        dem_path, rivers, rules.burn_depth_m, work / "burned.tif"
    )
    burned_m = np.where(river_cells, rules.burn_depth_m, 0.0)
    breached = whitebox.breach(
        binary, burned_path, work / "breached.tif", math.ceil(rules.breach_max_dist_m / cell)
    )
    pointer_path = whitebox.d8_pointer(binary, breached, work / "d8.tif")
    acc_path = whitebox.flow_accumulation(binary, pointer_path, work / "acc.tif")
    extracted = whitebox.extract_streams(binary, acc_path, work / "extracted.tif", threshold)
    streams_path = add_mapped_rivers(extracted, river_cells, work / "streams.tif")
    raw_hand = whitebox.elevation_above_stream(
        binary, breached, streams_path, work / "hand_raw.tif"
    )

    dem = _read(breached)
    land = np.isfinite(dem.values)
    pointer = np.nan_to_num(_read(pointer_path).values, nan=0.0).astype(np.int64)
    streams = np.nan_to_num(_read(streams_path).values, nan=0.0) > 0
    target = drainage_target(pointer, streams, land)
    restored, clipped = restore_burn(_read(raw_hand).values, burned_m, target)
    hand, sea_drained = fill_unreached(restored, dem.values + burned_m, land)
    outline = unary_union(list(districts.to_crs(epsg=epsg).geometry))
    regency = _regency_mask(outline, dem.transform, hand.shape) & land

    flow = Flow(pointer_path, pointer, _read(acc_path).values, land, dem.transform, epsg)
    basins, outlets, masks = _basins(region.rivers, flow, rules, binary, work)
    medians, (p5, p50, p95) = check_hand(hand, streams, land, regency, rules)

    vector = gpd.read_file(
        whitebox.streams_to_vector(binary, streams_path, pointer_path, work / "streams.shp")
    ).set_crs(epsg=epsg, allow_override=True)
    recall, precision = _osm_agreement(vector, waterways, rules, outline)

    hand_path = work / "hand.tif"
    _write(
        hand_path, np.where(np.isfinite(hand), hand, NODATA).astype(np.float32), dem.transform, epsg
    )
    edges = [0.0, *rules.distance_bands_m]
    report = HandReport(
        cell_size_m=cell,
        stream_threshold_km2=rules.stream_threshold_km2,
        stream_threshold_cells=threshold,
        basin_area_km2={
            r: round(a, 1) for r, a in zip(basins["river_id"], basins["area_km2"], strict=True)
        },
        regency_share_in_basin={
            river_id: round(float(mask[regency].mean()), 4) for river_id, mask in masks.items()
        },
        outlets=outlets,
        median_by_band_m={
            f"{low:.0f}-{high:.0f}": round(float(m), 2)
            for (low, high), m in zip(itertools.pairwise(edges), medians, strict=True)
        },
        hand_p5=round(p5, 2),
        hand_p50=round(p50, 2),
        hand_p95=round(p95, 2),
        sea_drained_cells=sea_drained,
        burned_river_km=round(float(rivers.length.sum()) / 1000.0, 1)
        if rules.burn_depth_m
        else 0.0,
        burn_clipped_cells=clipped,
        osm_river_recall=round(recall, 3),
        osm_precision=round(precision, 3),
    )
    return report, hand_path, vector.to_crs(WGS84)[["geometry"]], basins


def _basins(
    rivers: list[River], flow: Flow, rules: HandRules, binary: Path, work: Path
) -> tuple[gpd.GeoDataFrame, dict[str, tuple[float, float]], dict[str, BoolArray]]:
    snap_cells = math.ceil(rules.pour_point_snap_m / rules.cell_size_m)
    crs = f"EPSG:{flow.epsg}"
    rows: list[dict[str, object]] = []
    outlets: dict[str, tuple[float, float]] = {}
    masks: dict[str, BoolArray] = {}
    for river in rivers:
        lon, lat = river.glofas_point
        xy = gpd.GeoSeries.from_xy([lon], [lat], crs=WGS84).to_crs(crs).iloc[0]
        row, col = locate_outlet(
            flow.pointer, flow.accumulation, flow.land, flow.transform, (xy.x, xy.y), snap_cells
        )
        x, y = flow.transform @ (col + 0.5, row + 0.5)
        pour = work / f"outlet_{river.river_id}.shp"
        gpd.GeoDataFrame({"id": [1]}, geometry=gpd.points_from_xy([x], [y]), crs=crs).to_file(pour)
        basin = whitebox.watershed(
            binary, flow.pointer_path, pour, work / f"basin_{river.river_id}.tif"
        )
        mask = np.nan_to_num(_read(basin).values, nan=0.0) > 0
        if touches_edge(mask):
            raise ConfigurationError(
                f"basin of {river.river_id} touches the DEM edge; widen dem_bbox"
            )
        if any((mask & other).any() for other in masks.values()):
            raise ConfigurationError(f"basin of {river.river_id} overlaps another river's basin")
        masks[river.river_id] = mask
        outlet = gpd.GeoSeries.from_xy([x], [y], crs=crs).to_crs(WGS84).iloc[0]
        outlets[river.river_id] = (round(outlet.x, 5), round(outlet.y, 5))
        rows.append(
            {
                "river_id": river.river_id,
                "area_km2": float(mask.sum()) * rules.cell_size_m**2 / M2_PER_KM2,
                "geometry": basin_polygon(mask, flow.transform, flow.epsg).iloc[0],
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=WGS84), outlets, masks


def _osm_agreement(
    extracted: gpd.GeoDataFrame,
    waterways: gpd.GeoDataFrame,
    rules: HandRules,
    area: BaseGeometry,
) -> tuple[float, float]:
    streams = list(extracted.clip(area).geometry)
    rivers = list(waterways.loc[waterways["waterway"] == "river"].clip(area).geometry)
    recall, _ = stream_agreement(streams, rivers, rules.osm_match_m)
    _, precision = stream_agreement(streams, list(waterways.clip(area).geometry), rules.osm_match_m)
    if recall < rules.osm_min_recall:
        log.warning("only %.0f%% of OSM rivers lie near extracted streams", 100 * recall)
    return recall, precision


def _regency_mask(outline: BaseGeometry, transform: Affine, shape_: tuple[int, ...]) -> BoolArray:
    return geometry_mask([outline], out_shape=shape_, transform=transform, invert=True)


def _districts(out_dir: Path) -> gpd.GeoDataFrame:
    path = out_dir / "districts.parquet"
    if not path.is_file():
        raise InvalidInputError(f"missing {path.name}; build villages first")
    return gpd.read_parquet(path)


def _read(path: Path) -> Raster:
    with rasterio.open(path) as src:
        values = src.read(1, masked=True).astype(np.float64).filled(np.nan)
        return Raster(values, src.transform, int(src.crs.to_epsg()))


def _write(path: Path, values: FloatArray, transform: Affine, epsg: int) -> None:
    profile = {
        "driver": "GTiff",
        "height": values.shape[0],
        "width": values.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": f"EPSG:{epsg}",
        "transform": transform,
        "nodata": NODATA,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values.astype(np.float32), 1)


def _write_cog(source: Path, out: Path, work: Path) -> None:
    part = work / f"{out.stem}.cog{out.suffix}"
    rasterio.shutil.copy(source, part, driver="COG", compress="DEFLATE", predictor="YES")
    part.replace(out)
