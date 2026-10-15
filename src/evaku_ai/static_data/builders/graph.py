from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
from scipy.sparse import csgraph
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from evaku_ai.adapters.download import fetch
from evaku_ai.adapters.region import load_region
from evaku_ai.core.errors import InvalidInputError
from evaku_ai.domain.arrays import FloatArray
from evaku_ai.domain.rules import RoadNetworkRules
from evaku_ai.engine.graph import RoadGraph, to_csr
from evaku_ai.geo.crs import utm_epsg

log = logging.getLogger(__name__)

WGS84 = "EPSG:4326"
M_PER_S_PER_KMH = 1000 / 3600
M_PER_KM = 1000.0
MIN_LARGEST_COMPONENT_SHARE = 0.5
CLOSED_ACCESS = {"no", "private"}
EXCLUDED_SERVICE = {"parking_aisle", "driveway", "drive-through"}
EDGE_COLUMNS = [
    "edge_id",
    "u",
    "v",
    "highway_type",
    "is_bridge",
    "length_m",
    "travel_s",
    "osm_id",
    "geometry",
]
NODE_COLUMNS = ["node_id", "x_m", "y_m", "geometry"]
HSTORE_PAIR = re.compile(r'"((?:[^"\\]|\\.)*)"=>"((?:[^"\\]|\\.)*)"')

Coord = tuple[float, float]


@dataclass(frozen=True)
class GraphReport:
    nodes: int
    edges: int
    total_km: float
    bridges: int
    bridge_km: float
    components: int
    largest_component_share: float
    km_by_highway: dict[str, float]


def parse_tags(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    return {_unescape(k): _unescape(v) for k, v in HSTORE_PAIR.findall(raw)}


def read_osm_lines(path: Path, bbox: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    columns = ["osm_id", "highway", "other_tags"]
    lines = gpd.read_file(path, layer="lines", bbox=bbox, columns=columns)
    return lines.set_crs(WGS84, allow_override=True)


def select_roads(lines: gpd.GeoDataFrame, rules: RoadNetworkRules) -> gpd.GeoDataFrame:
    tags = lines["other_tags"].fillna("").astype(str).map(parse_tags)
    highway = lines["highway"].fillna("").astype(str)
    usable = highway.map(lambda h: rules.speed_kmh(h) is not None) & tags.map(_open_to_vehicles)
    roads = lines.loc[usable, ["osm_id", "geometry"]].copy()
    roads["highway_type"] = highway[usable]
    roads["is_bridge"] = tags[usable].map(lambda t: t.get("bridge", "no") != "no")
    return roads


def region_boundary(districts: gpd.GeoDataFrame, buffer_m: float, epsg: int) -> BaseGeometry:
    projected = districts.to_crs(epsg=epsg).geometry
    buffered = gpd.GeoSeries([unary_union(list(projected)).buffer(buffer_m)], crs=projected.crs)
    return buffered.to_crs(WGS84).iloc[0]


def clip_roads(roads: gpd.GeoDataFrame, boundary: BaseGeometry) -> gpd.GeoDataFrame:
    clipped = roads.copy()
    clipped["geometry"] = roads.geometry.intersection(boundary)
    clipped = clipped.explode(index_parts=False)
    keep = (clipped.geom_type == "LineString") & ~clipped.is_empty
    return clipped[keep].reset_index(drop=True)


def split_at_junctions(lines: Sequence[LineString]) -> list[tuple[int, LineString]]:
    coords = [[(float(x), float(y)) for x, y in line.coords] for line in lines]
    uses = Counter(c for line in coords for c in set(line))
    ends = {c for line in coords for c in (line[0], line[-1])}
    nodes = {c for c, n in uses.items() if n > 1} | ends
    segments: list[tuple[int, LineString]] = []
    for index, line in enumerate(coords):
        current = [line[0]]
        for coord in line[1:]:
            current.append(coord)
            if coord in nodes:
                if len(set(current)) > 1:
                    segments.append((index, LineString(current)))
                current = [coord]
    return segments


def travel_seconds(
    length_m: FloatArray, highway: Sequence[str], rules: RoadNetworkRules
) -> FloatArray:
    speeds = [rules.speed_kmh(h) for h in highway]
    unknown = sorted({h for h, s in zip(highway, speeds, strict=True) if s is None})
    if unknown:
        raise InvalidInputError(f"no speed configured for highway types {unknown}")
    speed_m_s = np.array(speeds, dtype=np.float64) * M_PER_S_PER_KMH
    return np.asarray(length_m, dtype=np.float64) / speed_m_s


def assemble(
    roads: gpd.GeoDataFrame,
    segments: Sequence[tuple[int, LineString]],
    epsg: int,
    rules: RoadNetworkRules,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    starts = [_first(seg) for _, seg in segments]
    ends = [_last(seg) for _, seg in segments]
    coords = sorted(set(starts) | set(ends))
    node_index = {c: i for i, c in enumerate(coords)}
    u = np.array([node_index[c] for c in starts], dtype=np.int64)
    v = np.array([node_index[c] for c in ends], dtype=np.int64)
    source = [i for i, _ in segments]
    edges = gpd.GeoDataFrame(
        {
            "u": u,
            "v": v,
            "highway_type": roads["highway_type"].to_numpy()[source],
            "is_bridge": roads["is_bridge"].to_numpy(dtype=bool)[source],
            "osm_id": roads["osm_id"].astype(str).to_numpy()[source],
        },
        geometry=[seg for _, seg in segments],
        crs=WGS84,
    )
    edges = edges[edges["u"] != edges["v"]].reset_index(drop=True)
    edges["length_m"] = edges.to_crs(epsg=epsg).length.to_numpy()
    edges["travel_s"] = travel_seconds(
        edges["length_m"].to_numpy(), list(edges["highway_type"]), rules
    )
    edges["edge_id"] = np.arange(len(edges), dtype=np.int64)

    points = gpd.GeoSeries([Point(c) for c in coords], crs=WGS84)
    projected = points.to_crs(epsg=epsg)
    nodes = gpd.GeoDataFrame(
        {
            "node_id": np.arange(len(coords), dtype=np.int64),
            "x_m": projected.x.to_numpy(),
            "y_m": projected.y.to_numpy(),
        },
        geometry=points,
        crs=WGS84,
    )
    return nodes[NODE_COLUMNS], edges[EDGE_COLUMNS]


def to_road_graph(nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame) -> RoadGraph:
    return RoadGraph(
        edge_u=edges["u"].to_numpy(dtype=np.int64),
        edge_v=edges["v"].to_numpy(dtype=np.int64),
        travel_s=edges["travel_s"].to_numpy(dtype=np.float64),
        length_m=edges["length_m"].to_numpy(dtype=np.float64),
        hand_min=np.full(len(edges), np.nan),
        is_bridge=edges["is_bridge"].to_numpy(dtype=bool),
        node_x=nodes["x_m"].to_numpy(dtype=np.float64),
        node_y=nodes["y_m"].to_numpy(dtype=np.float64),
    )


def check_network(nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame) -> GraphReport:
    if nodes.empty or edges.empty:
        raise InvalidInputError("road network is empty")
    if not np.array_equal(nodes["node_id"].to_numpy(), np.arange(len(nodes))):
        raise InvalidInputError("node_id must run 0..N-1")
    if not np.array_equal(edges["edge_id"].to_numpy(), np.arange(len(edges))):
        raise InvalidInputError("edge_id must run 0..M-1")
    if not (edges["length_m"] > 0).all() or not np.isfinite(edges["length_m"]).all():
        raise InvalidInputError("length_m must be finite and > 0")
    if not (edges.geom_type == "LineString").all() or not edges.is_valid.all():
        raise InvalidInputError("edge geometries must be valid LineStrings")
    graph = to_road_graph(nodes, edges)
    n_components, labels = csgraph.connected_components(to_csr(graph), directed=False)
    largest = float(np.bincount(labels).max() / graph.n_nodes)
    if largest <= MIN_LARGEST_COMPONENT_SHARE:
        raise InvalidInputError(f"largest connected component holds only {largest:.0%} of nodes")
    km = edges.groupby("highway_type")["length_m"].sum() / M_PER_KM
    bridges = edges[edges["is_bridge"]]
    return GraphReport(
        nodes=len(nodes),
        edges=len(edges),
        total_km=round(float(edges["length_m"].sum() / M_PER_KM), 1),
        bridges=len(bridges),
        bridge_km=round(float(bridges["length_m"].sum() / M_PER_KM), 2),
        components=int(n_components),
        largest_component_share=round(largest, 4),
        km_by_highway={
            str(k): round(float(v), 1) for k, v in km.sort_values(ascending=False).items()
        },
    )


def build(region_path: Path, out_dir: Path = Path("data/processed")) -> GraphReport:
    region = load_region(region_path)
    rules = region.road_network()
    epsg = utm_epsg(*region.center)
    districts_path = out_dir / "districts.parquet"
    if not districts_path.is_file():
        raise InvalidInputError(f"{districts_path} not found; build the village boundaries first")
    boundary = region_boundary(gpd.read_parquet(districts_path), rules.buffer_m, epsg)

    source = fetch(region.inputs.osm_extract)
    min_x, min_y, max_x, max_y = (float(b) for b in boundary.bounds)
    lines = read_osm_lines(source, (min_x, min_y, max_x, max_y))
    roads = clip_roads(select_roads(lines, rules), boundary)
    roads = roads.sort_values("osm_id", key=lambda s: s.astype(np.int64)).reset_index(drop=True)
    segments = split_at_junctions(list(roads.geometry))
    nodes, edges = assemble(roads, segments, epsg, rules)
    report = check_network(nodes, edges)

    out_dir.mkdir(parents=True, exist_ok=True)
    nodes.to_parquet(out_dir / "road_nodes.parquet")
    edges.to_parquet(out_dir / "road_segments.parquet")
    remote = region.inputs.osm_extract
    meta = {
        "region_id": region.region_id,
        "source": remote.source,
        "license": remote.license,
        "url": str(remote.url),
        "sha256": remote.sha256,
        "utm_epsg": epsg,
        "buffer_m": rules.buffer_m,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **asdict(report),
    }
    (out_dir / "graph.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info(
        "road graph: %d nodes, %d edges, %.1f km, %d bridges, largest component %.1f%%",
        report.nodes,
        report.edges,
        report.total_km,
        report.bridges,
        100 * report.largest_component_share,
    )
    return report


def _open_to_vehicles(tags: dict[str, str]) -> bool:
    if tags.get("access") in CLOSED_ACCESS or tags.get("motor_vehicle") in CLOSED_ACCESS:
        return False
    return tags.get("area") != "yes" and tags.get("service") not in EXCLUDED_SERVICE


def _unescape(text: str) -> str:
    return text.replace('\\"', '"').replace("\\\\", "\\")


def _first(line: LineString) -> Coord:
    x, y = line.coords[0]
    return float(x), float(y)


def _last(line: LineString) -> Coord:
    x, y = line.coords[-1]
    return float(x), float(y)
