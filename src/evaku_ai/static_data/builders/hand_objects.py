from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from evaku_ai.adapters.download import fetch, file_sha256
from evaku_ai.adapters.region import load_region
from evaku_ai.core.errors import InvalidInputError
from evaku_ai.domain.arrays import BoolArray, FloatArray, IntArray
from evaku_ai.domain.region import RegionConfig
from evaku_ai.domain.rules import HandRules
from evaku_ai.geo.crs import utm_epsg
from evaku_ai.geo.raster import GridTransform, line_samples, reduce_by_line, sample_values
from evaku_ai.geo.vector import crossing_points, near_crossings, weighted_median
from evaku_ai.static_data.builders.population import (
    node_villages,
    pixels_in_villages,
    read_window,
    snap_by_village,
)

log = logging.getLogger(__name__)

WGS84 = "EPSG:4326"
INPUTS = (
    "hand.tif",
    "hand.meta.json",
    "river_basins.parquet",
    "streams.parquet",
    "road_nodes.parquet",
    "road_segments.parquet",
    "facilities.parquet",
    "villages.parquet",
)


@dataclass(frozen=True)
class HandGrid:
    values: FloatArray
    transform: GridTransform
    footprint: BaseGeometry


@dataclass(frozen=True)
class HandObjectsReport:
    edges: int
    edges_without_hand: int
    edges_with_skipped_crossings: int
    edges_at_zero: int
    edge_hand_p10_p50_p90: tuple[float, float, float]
    edge_hand_median_near_streams: float
    edge_hand_median_far_from_streams: float
    node_hand_p10_p50_p90: tuple[float, float, float]
    facility_hand_min: float
    lowest_facilities: list[str]
    objects_per_river: dict[str, dict[str, int]]
    population_at_or_below_m: dict[str, float]


def read_hand(path: Path, dem_bbox: tuple[float, float, float, float], epsg: int) -> HandGrid:
    with rasterio.open(path) as src:
        values = src.read(1, masked=True).astype(np.float64).filled(np.nan)
        t = src.transform
    footprint = gpd.GeoSeries([box(*dem_bbox)], crs=WGS84).to_crs(epsg=epsg).iloc[0]
    return HandGrid(values, (t.a, t.b, t.c, t.d, t.e, t.f), footprint)


def sample_hand(grid: HandGrid, x: FloatArray, y: FloatArray) -> FloatArray:
    hand = sample_values(grid.values, grid.transform, x, y, None)
    # Inside the DEM, no data means the masked sea or sea-level ponds: the point is at sea level.
    at_sea = ~np.isfinite(hand) & shapely.contains_xy(grid.footprint, x, y)
    return np.where(at_sea, 0.0, hand)


def edge_hand(
    grid: HandGrid, lines: list[BaseGeometry], streams: list[BaseGeometry], rules: HandRules
) -> tuple[FloatArray, IntArray]:
    x, y, line = line_samples(lines, rules.sample_step_m)
    cross_x, cross_y, cross_line = crossing_points(lines, streams)
    skip = near_crossings(x, y, line, cross_x, cross_y, cross_line, rules.crossing_buffer_m)
    hand = sample_hand(grid, x, y)
    lowest = reduce_by_line(np.where(skip, np.nan, hand), line, len(lines), "min")
    # A short road lying entirely inside a crossing buffer keeps its plain minimum.
    plain = reduce_by_line(hand, line, len(lines), "min")
    lowest = np.where(np.isfinite(lowest), lowest, plain)
    crossings = np.bincount(cross_line, minlength=len(lines)).astype(np.int64)
    return lowest, crossings


def pixel_hand(
    region: RegionConfig,
    villages: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    grid: HandGrid,
    epsg: int,
) -> gpd.GeoDataFrame:
    bbox = tuple(float(v) for v in villages.total_bounds)
    fine = read_window(
        fetch(region.inputs.population.total_100m), (bbox[0], bbox[1], bbox[2], bbox[3])
    )
    pixels = pixels_in_villages(fine, villages)
    node_id, _ = snap_by_village(pixels, nodes, node_villages(nodes, villages), epsg)
    projected = pixels.to_crs(epsg=epsg)
    pixels["node_id"] = node_id
    pixels["hand"] = sample_hand(
        grid, projected.geometry.x.to_numpy(), projected.geometry.y.to_numpy()
    )
    return pixels


def node_hand(grid: HandGrid, nodes: gpd.GeoDataFrame, pixels: gpd.GeoDataFrame) -> pd.DataFrame:
    at_node = sample_hand(grid, nodes["x_m"].to_numpy(), nodes["y_m"].to_numpy())
    people = weighted_median(
        pixels["hand"].to_numpy(),
        pixels["population"].to_numpy(),
        pixels["node_id"].to_numpy(dtype=np.int64),
        len(nodes),
    )
    return pd.DataFrame(
        {
            "node_id": nodes["node_id"].to_numpy(dtype=np.int64),
            "hand": np.where(np.isfinite(people), people, at_node),
            "hand_at_node": at_node,
        }
    )


def village_hand(
    pixels: gpd.GeoDataFrame, village_ids: list[str], rules: HandRules
) -> pd.DataFrame:
    index = {v: i for i, v in enumerate(village_ids)}
    group = pixels["village_id"].map(index).to_numpy(dtype=np.int64)
    hand = pixels["hand"].to_numpy()
    people = pixels["population"].to_numpy()
    table = pd.DataFrame(
        {
            "village_id": village_ids,
            "hand_median": weighted_median(hand, people, group, len(village_ids)),
        }
    )
    for limit in rules.village_low_hand_m:
        low = np.bincount(group, weights=people * (hand <= limit), minlength=len(village_ids))
        table[_low_column(limit)] = low
    return table


def assign_river(points: gpd.GeoSeries, basins: gpd.GeoDataFrame) -> pd.Series[str]:
    frame = gpd.GeoDataFrame(geometry=points.reset_index(drop=True))
    joined = gpd.sjoin(frame, basins[["river_id", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated()]
    return joined["river_id"].reindex(frame.index)


def check_objects(
    edges: pd.DataFrame,
    edge_inside: BoolArray,
    nodes: pd.DataFrame,
    facilities: pd.DataFrame,
    stream_distance_m: FloatArray,
    rules: HandRules,
) -> tuple[float, float]:
    missing = edge_inside & ~np.isfinite(edges["hand_min"].to_numpy())
    if missing.any():
        raise InvalidInputError(f"{int(missing.sum())} road edges inside the DEM have no HAND")
    if not np.isfinite(facilities["hand"].to_numpy()).all():
        raise InvalidInputError("every facility needs a HAND value")
    for label, values in (
        ("edge hand_min", edges["hand_min"]),
        ("node hand", nodes["hand"]),
        ("facility hand", facilities["hand"]),
    ):
        if (values.dropna() < 0).any():
            raise InvalidInputError(f"negative {label}")
    near_m, far_m = rules.distance_bands_m[0], rules.distance_bands_m[1]
    hand_min = edges["hand_min"].to_numpy()
    near = float(np.nanmedian(hand_min[stream_distance_m <= near_m]))
    far = float(np.nanmedian(hand_min[stream_distance_m > far_m]))
    if not near < far:
        raise InvalidInputError(
            f"roads near streams should sit lower: median {near:.1f} m vs {far:.1f} m"
        )
    return near, far


def build(region_path: Path, out_dir: Path = Path("data/processed")) -> HandObjectsReport:
    started = time.monotonic()
    region = load_region(region_path)
    rules = region.hand_rules()
    epsg = utm_epsg(*region.center)
    missing = [name for name in INPUTS if not (out_dir / name).is_file()]
    if missing:
        raise InvalidInputError(
            f"missing {missing}; build villages, graph, facilities, and HAND first"
        )
    grid = read_hand(out_dir / "hand.tif", region.dem_bbox, epsg)
    basins = gpd.read_parquet(out_dir / "river_basins.parquet").to_crs(epsg=epsg)
    streams = list(gpd.read_parquet(out_dir / "streams.parquet").to_crs(epsg=epsg).geometry)
    nodes = gpd.read_parquet(out_dir / "road_nodes.parquet")
    segments = gpd.read_parquet(out_dir / "road_segments.parquet")
    facilities = gpd.read_parquet(out_dir / "facilities.parquet")
    villages = gpd.read_parquet(out_dir / "villages.parquet")

    lines = list(segments.to_crs(epsg=epsg).geometry)
    hand_min, crossings = edge_hand(grid, lines, streams, rules)
    midpoints = gpd.GeoSeries(shapely.line_interpolate_point(lines, 0.5, normalized=True), crs=epsg)
    edges = pd.DataFrame(
        {
            "edge_id": segments["edge_id"].to_numpy(dtype=np.int64),
            "hand_min": hand_min,
            "river_id": assign_river(midpoints, basins).to_numpy(),
            "crossings": crossings,
        }
    )

    pixels = pixel_hand(region, villages, nodes, grid, epsg)
    node_table = node_hand(grid, nodes, pixels)
    node_points = gpd.GeoSeries.from_xy(nodes["x_m"], nodes["y_m"], crs=epsg)
    node_table["river_id"] = assign_river(node_points, basins).to_numpy()

    facility_points = facilities.to_crs(epsg=epsg).geometry
    facility_table = pd.DataFrame(
        {
            "facility_id": facilities["facility_id"].to_numpy(dtype=np.int64),
            "hand": sample_hand(grid, facility_points.x.to_numpy(), facility_points.y.to_numpy()),
            "river_id": assign_river(facility_points, basins).to_numpy(),
        }
    )
    village_table = village_hand(pixels, list(villages["village_id"]), rules)

    stream_tree = shapely.STRtree(np.asarray(streams, dtype=object))
    _, distance = stream_tree.query_nearest(
        np.asarray(lines, dtype=object), return_distance=True, all_matches=False
    )
    near, far = check_objects(
        edges,
        shapely.contains_xy(grid.footprint, *shapely.get_coordinates(midpoints.to_numpy()).T),
        node_table,
        facility_table,
        distance,
        rules,
    )
    report = _report(edges, node_table, facility_table, facilities, pixels, near, far, rules)

    out_dir.mkdir(parents=True, exist_ok=True)
    edges.to_parquet(out_dir / "edge_hand.parquet")
    node_table.to_parquet(out_dir / "node_hand.parquet")
    facility_table.to_parquet(out_dir / "facility_hand.parquet")
    village_table.to_parquet(out_dir / "village_hand.parquet")
    hand_meta = json.loads((out_dir / "hand.meta.json").read_text(encoding="utf-8"))
    meta = {
        "region_id": region.region_id,
        "hand_sha256": file_sha256(out_dir / "hand.tif"),
        "hand_built_at": hand_meta.get("built_at"),
        "rules": {
            "sample_step_m": rules.sample_step_m,
            "crossing_buffer_m": rules.crossing_buffer_m,
            "village_low_hand_m": list(rules.village_low_hand_m),
        },
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seconds": round(time.monotonic() - started, 1),
        **asdict(report),
    }
    (out_dir / "hand_objects.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info(
        "edge hand_min median near streams %.1f m vs far %.1f m; %d edges skipped crossings",
        near,
        far,
        report.edges_with_skipped_crossings,
    )
    return report


def _report(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    facility_table: pd.DataFrame,
    facilities: gpd.GeoDataFrame,
    pixels: gpd.GeoDataFrame,
    near: float,
    far: float,
    rules: HandRules,
) -> HandObjectsReport:
    hand_min = edges["hand_min"].to_numpy()
    lowest = facility_table.nsmallest(5, "hand")
    rows = facilities.set_index("facility_id").loc[lowest["facility_id"]]
    # Many OSM and Overture places have no name; the type still tells what the place is.
    names = [
        f"{name or kind} #{facility_id}"
        for facility_id, name, kind in zip(rows.index, rows["name"], rows["type"], strict=True)
    ]
    per_river: dict[str, dict[str, int]] = {}
    for label, table in (("edges", edges), ("nodes", nodes), ("facilities", facility_table)):
        for river_id, count in table["river_id"].fillna("none").value_counts().items():
            per_river.setdefault(str(river_id), {})[label] = int(count)
    people = pixels["population"].to_numpy()
    return HandObjectsReport(
        edges=len(edges),
        edges_without_hand=int((~np.isfinite(hand_min)).sum()),
        edges_with_skipped_crossings=int((edges["crossings"] > 0).sum()),
        edges_at_zero=int((hand_min == 0).sum()),
        edge_hand_p10_p50_p90=_percentiles(hand_min),
        edge_hand_median_near_streams=round(near, 2),
        edge_hand_median_far_from_streams=round(far, 2),
        node_hand_p10_p50_p90=_percentiles(nodes["hand"].to_numpy()),
        facility_hand_min=round(float(facility_table["hand"].min()), 2),
        lowest_facilities=[
            f"{name} ({hand:.1f} m)" for name, hand in zip(names, lowest["hand"], strict=True)
        ],
        objects_per_river=per_river,
        population_at_or_below_m={
            f"{limit:g}": round(float(people[pixels["hand"].to_numpy() <= limit].sum()), 1)
            for limit in rules.village_low_hand_m
        },
    )


def _percentiles(values: FloatArray) -> tuple[float, float, float]:
    p10, p50, p90 = (round(float(v), 2) for v in np.nanpercentile(values, [10, 50, 90]))
    return p10, p50, p90


def _low_column(limit: float) -> str:
    return f"population_hand_le_{limit:g}m"
