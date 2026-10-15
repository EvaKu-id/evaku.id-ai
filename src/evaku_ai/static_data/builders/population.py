from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

from evaku_ai.adapters.download import fetch
from evaku_ai.adapters.region import load_region
from evaku_ai.core.errors import InvalidInputError
from evaku_ai.domain.arrays import FloatArray, IntArray
from evaku_ai.domain.region import RemoteFile
from evaku_ai.engine.graph import to_csr
from evaku_ai.geo.crs import utm_epsg
from evaku_ai.geo.raster import GridTransform, pixel_centers, sample_at
from evaku_ai.static_data.builders.graph import to_road_graph

log = logging.getLogger(__name__)

WGS84 = "EPSG:4326"
TOTAL_TOLERANCE = 1e-6
COARSE_MARGIN_DEG = 0.02
NODE_COLUMNS = ["node_id", "village_id", "population", "under5", "elderly"]
VILLAGE_COLUMNS = ["village_id", "population", "children_under5", "elderly"]


@dataclass(frozen=True)
class Grid:
    values: FloatArray
    transform: GridTransform
    nodata: float | None


@dataclass(frozen=True)
class PopulationReport:
    total: float
    under5: float
    elderly: float
    under5_share: float
    elderly_share: float
    villages: int
    villages_without_nodes: list[str]
    outside_main_component_share: float
    max_village: str
    min_village: str


def read_window(path: Path, bbox: tuple[float, float, float, float]) -> Grid:
    with rasterio.open(path) as src:
        window = from_bounds(*bbox, transform=src.transform).round_offsets().round_lengths()
        values = src.read(1, window=window, boundless=True, fill_value=0).astype(np.float64)
        t = src.window_transform(window)
        return Grid(values, (t.a, t.b, t.c, t.d, t.e, t.f), src.nodata)


def pixels_in_villages(grid: Grid, villages: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    x, y, population = pixel_centers(grid.values, grid.transform, grid.nodata)
    points = gpd.GeoDataFrame(
        {"x": x, "y": y, "population": population},
        geometry=gpd.points_from_xy(x, y),
        crs=WGS84,
    )
    joined = gpd.sjoin(points, villages[["village_id", "geometry"]], predicate="within")
    joined = joined[~joined.index.duplicated()]
    return joined.drop(columns="index_right").reset_index(drop=True)


def age_shares(
    pixels: gpd.GeoDataFrame, total: Grid, groups: dict[str, list[Grid]]
) -> dict[str, FloatArray]:
    x, y = pixels["x"].to_numpy(), pixels["y"].to_numpy()
    coarse_total = sample_at(total.values, total.transform, x, y, total.nodata)
    weights = pixels["population"].to_numpy()
    shares: dict[str, FloatArray] = {}
    for name, bands in groups.items():
        counted = sum(sample_at(b.values, b.transform, x, y, b.nodata) for b in bands)
        share = np.divide(
            counted, coarse_total, out=np.full(x.shape[0], np.nan), where=coarse_total > 0
        )
        known = np.isfinite(share)
        # Cells without coarse data borrow the population-weighted share of the whole regency.
        fallback = float(np.average(share[known], weights=weights[known])) if known.any() else 0.0
        shares[name] = np.clip(np.where(known, share, fallback), 0.0, 1.0)
    return shares


def node_villages(nodes: gpd.GeoDataFrame, villages: gpd.GeoDataFrame) -> pd.Series[str]:
    joined = gpd.sjoin(
        nodes[["node_id", "geometry"]],
        villages[["village_id", "geometry"]],
        how="left",
        predicate="within",
    )
    joined = joined[~joined.index.duplicated()]
    return joined["village_id"].reindex(nodes.index)


def snap_by_village(
    pixels: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    node_village: pd.Series[str],
    epsg: int,
) -> tuple[IntArray, list[str]]:
    projected = pixels.to_crs(epsg=epsg)
    pixel_xy = np.c_[projected.geometry.x, projected.geometry.y]
    node_xy = np.c_[nodes["x_m"].to_numpy(), nodes["y_m"].to_numpy()]
    node_ids = nodes["node_id"].to_numpy(dtype=np.int64)
    everywhere = cKDTree(node_xy)
    assigned = np.empty(len(pixels), dtype=np.int64)
    without_nodes: list[str] = []
    for village_id, rows in pixels.groupby("village_id").groups.items():
        index = np.asarray(rows)
        own = np.flatnonzero((node_village == village_id).to_numpy())
        if own.size:
            _, nearest = cKDTree(node_xy[own]).query(pixel_xy[index])
            assigned[index] = node_ids[own[nearest]]
        else:
            without_nodes.append(str(village_id))
            _, nearest = everywhere.query(pixel_xy[index])
            assigned[index] = node_ids[nearest]
    return assigned, sorted(without_nodes)


def aggregate(
    pixels: gpd.GeoDataFrame,
    node_id: IntArray,
    shares: dict[str, FloatArray],
    nodes: gpd.GeoDataFrame,
    node_village: pd.Series[str],
    village_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    population = pixels["population"].to_numpy()
    under5 = population * shares["under5"]
    elderly = population * shares["elderly"]
    n_nodes = len(nodes)
    node_table = pd.DataFrame(
        {
            "node_id": nodes["node_id"].to_numpy(dtype=np.int64),
            "village_id": node_village.to_numpy(),
            "population": np.bincount(node_id, weights=population, minlength=n_nodes),
            "under5": np.bincount(node_id, weights=under5, minlength=n_nodes),
            "elderly": np.bincount(node_id, weights=elderly, minlength=n_nodes),
        }
    )
    per_village = (
        pd.DataFrame(
            {
                "village_id": pixels["village_id"],
                "population": population,
                "children_under5": under5,
                "elderly": elderly,
            }
        )
        .groupby("village_id")
        .sum()
        .reindex(village_ids, fill_value=0.0)
        .reset_index()
    )
    return node_table[NODE_COLUMNS], per_village[VILLAGE_COLUMNS]


def check_population(
    raster_total: float,
    node_table: pd.DataFrame,
    village_table: pd.DataFrame,
    without_nodes: list[str],
    outside_main_share: float,
) -> PopulationReport:
    for label, total in (
        ("nodes", node_table["population"].sum()),
        ("villages", village_table["population"].sum()),
    ):
        if abs(total - raster_total) > TOTAL_TOLERANCE * max(raster_total, 1.0):
            raise InvalidInputError(
                f"population on {label} {total:.1f} != raster total {raster_total:.1f}"
            )
    empty = village_table.loc[village_table["population"] <= 0, "village_id"].tolist()
    if empty:
        raise InvalidInputError(f"villages without population: {empty}")
    for column in ("children_under5", "elderly"):
        if (village_table[column] > village_table["population"] + TOTAL_TOLERANCE).any():
            raise InvalidInputError(f"{column} exceeds population")
    ordered = village_table.sort_values("population")
    under5, elderly = (
        float(village_table["children_under5"].sum()),
        float(village_table["elderly"].sum()),
    )
    return PopulationReport(
        total=round(raster_total, 1),
        under5=round(under5, 1),
        elderly=round(elderly, 1),
        under5_share=round(under5 / raster_total, 4),
        elderly_share=round(elderly / raster_total, 4),
        villages=len(village_table),
        villages_without_nodes=without_nodes,
        outside_main_component_share=round(outside_main_share, 4),
        max_village=str(ordered["village_id"].iloc[-1]),
        min_village=str(ordered["village_id"].iloc[0]),
    )


def build(region_path: Path, out_dir: Path = Path("data/processed")) -> PopulationReport:
    region = load_region(region_path)
    source = region.inputs.population
    epsg = utm_epsg(*region.center)
    villages, nodes, edges = _static_layers(out_dir)
    min_x, min_y, max_x, max_y = (float(v) for v in villages.total_bounds)
    bbox = (min_x, min_y, max_x, max_y)
    coarse_bbox = (
        min_x - COARSE_MARGIN_DEG,
        min_y - COARSE_MARGIN_DEG,
        max_x + COARSE_MARGIN_DEG,
        max_y + COARSE_MARGIN_DEG,
    )

    fine = read_window(fetch(source.total_100m), bbox)
    pixels = pixels_in_villages(fine, villages)
    coarse = read_window(fetch(source.total_1km), coarse_bbox)
    groups = {
        "under5": [read_window(fetch(r), coarse_bbox) for r in source.under5_1km],
        "elderly": [read_window(fetch(r), coarse_bbox) for r in source.elderly_1km],
    }
    shares = age_shares(pixels, coarse, groups)
    village_of_node = node_villages(nodes, villages)
    node_id, without_nodes = snap_by_village(pixels, nodes, village_of_node, epsg)
    node_table, village_table = aggregate(
        pixels, node_id, shares, nodes, village_of_node, list(villages["village_id"])
    )
    report = check_population(
        float(pixels["population"].sum()),
        node_table,
        village_table,
        without_nodes,
        _outside_main_share(node_table, nodes, edges),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    node_table.to_parquet(out_dir / "node_population.parquet")
    village_table.to_parquet(out_dir / "village_population.parquet")
    files: list[RemoteFile] = [
        source.total_100m,
        source.total_1km,
        *source.under5_1km,
        *source.elderly_1km,
    ]
    meta = {
        "region_id": region.region_id,
        "year": source.year,
        "source": source.total_100m.source,
        "license": source.total_100m.license,
        "files": {f.path.name: f.sha256 for f in files},
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **asdict(report),
    }
    (out_dir / "population.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info(
        "population %.0f (under 5: %.1f%%, 60+: %.1f%%) across %d villages",
        report.total,
        100 * report.under5_share,
        100 * report.elderly_share,
        report.villages,
    )
    return report


def _static_layers(out_dir: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    names = ("villages", "road_nodes", "road_segments")
    paths = [out_dir / f"{name}.parquet" for name in names]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise InvalidInputError(f"missing {missing}; build villages and the road graph first")
    villages, nodes, edges = (gpd.read_parquet(p) for p in paths)
    return villages, nodes, edges


def _outside_main_share(
    node_table: pd.DataFrame, nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame
) -> float:
    graph = to_road_graph(nodes, edges)
    _, labels = csgraph.connected_components(to_csr(graph), directed=False)
    main = labels == np.bincount(labels).argmax()
    people = node_table["population"].to_numpy()
    return float(people[~main].sum() / people.sum()) if people.sum() > 0 else 0.0
