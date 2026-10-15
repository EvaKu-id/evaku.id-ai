import copy
import json

import geopandas as gpd
import numpy as np
import pytest
from pyproj import Geod
from shapely.geometry import LineString, box

from evaku_ai.core.errors import InvalidInputError
from evaku_ai.domain.rules import RoadNetworkRules
from evaku_ai.static_data.builders import graph as builder
from tests.fixtures.region import VALID

EPSG = 32647
RULES = RoadNetworkRules()
NODES = {
    1: (98.000, 4.300),
    2: (98.010, 4.300),
    3: (98.020, 4.300),
    4: (98.010, 4.290),
    5: (98.010, 4.310),
    6: (98.020, 4.310),
    7: (98.015, 4.295),
    8: (98.015, 4.305),
    9: (98.005, 4.305),
    10: (98.030, 4.300),
    11: (98.000, 4.290),
    12: (98.050, 4.350),
    13: (98.051, 4.350),
    14: (98.010, 4.320),
    15: (98.012, 4.322),
}
WAYS = [
    (101, [1, 2, 3], {"highway": "primary"}),
    (102, [4, 2, 5], {"highway": "residential"}),
    (103, [3, 6], {"highway": "primary", "bridge": "yes"}),
    (104, [7, 8], {"highway": "residential", "bridge": "yes", "layer": "1"}),
    (105, [9, 2], {"highway": "footway"}),
    (106, [3, 10], {"highway": "residential", "access": "private"}),
    (107, [1, 11], {"highway": "service", "service": "driveway"}),
    (108, [12, 13], {"highway": "residential"}),
    (109, [5, 14, 15, 14], {"highway": "track"}),
]


def write_osm(path):
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<osm version="0.6" generator="test">']
    for node_id, (lon, lat) in NODES.items():
        lines.append(f'<node id="{node_id}" version="1" lat="{lat}" lon="{lon}"/>')
    for way_id, refs, tags in WAYS:
        lines.append(f'<way id="{way_id}" version="1">')
        lines += [f'<nd ref="{ref}"/>' for ref in refs]
        lines += [f'<tag k="{k}" v="{v}"/>' for k, v in tags.items()]
        lines.append("</way>")
    lines.append("</osm>")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@pytest.fixture
def osm(tmp_path):
    return write_osm(tmp_path / "region.osm")


@pytest.fixture
def roads(osm):
    lines = builder.read_osm_lines(osm, (97.9, 4.2, 98.1, 4.4))
    return builder.select_roads(lines, RULES).sort_values("osm_id").reset_index(drop=True)


def network(roads):
    segments = builder.split_at_junctions(list(roads.geometry))
    return builder.assemble(roads, segments, EPSG, RULES)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, {}),
        ("", {}),
        ('"bridge"=>"yes"', {"bridge": "yes"}),
        ('"access"=>"no","layer"=>"1"', {"access": "no", "layer": "1"}),
        ('"name"=>"Jalan \\"Lama\\", Km 3"', {"name": 'Jalan "Lama", Km 3'}),
    ],
)
def test_parse_tags(raw, expected):
    assert builder.parse_tags(raw) == expected


def test_read_osm_lines_filters_bbox(osm):
    lines = builder.read_osm_lines(osm, (97.99, 4.28, 98.04, 4.33))
    assert "108" not in set(lines["osm_id"])
    assert {"osm_id", "highway", "other_tags", "geometry"} <= set(lines.columns)


def test_select_roads_keeps_vehicle_roads(roads):
    assert list(roads["osm_id"]) == ["101", "102", "103", "104", "108", "109"]
    assert dict(zip(roads["osm_id"], roads["is_bridge"], strict=True)) == {
        "101": False,
        "102": False,
        "103": True,
        "104": True,
        "108": False,
        "109": False,
    }


def test_split_at_shared_nodes_only(roads):
    segments = builder.split_at_junctions(list(roads.geometry))
    by_way = {}
    for index, segment in segments:
        by_way.setdefault(roads.loc[index, "osm_id"], []).append(segment)
    assert len(by_way["101"]) == 2
    assert len(by_way["102"]) == 2
    assert len(by_way["104"]) == 1
    assert len(by_way["109"]) == 2


def test_assemble_ids_lengths_and_loops(roads):
    nodes, edges = network(roads)
    assert list(edges["edge_id"]) == list(range(8))
    assert list(nodes["node_id"]) == list(range(11))
    assert (edges["u"] != edges["v"]).all()
    assert list(edges.columns) == builder.EDGE_COLUMNS
    first = edges[edges["osm_id"] == "101"].iloc[0]
    geodesic = Geod(ellps="WGS84").line_length([98.000, 98.010], [4.300, 4.300])
    assert first["length_m"] == pytest.approx(geodesic, rel=1e-3)
    assert first["travel_s"] == pytest.approx(first["length_m"] / (50 / 3.6))


def test_assemble_is_deterministic(roads):
    nodes_a, edges_a = network(roads)
    nodes_b, edges_b = network(roads)
    assert edges_a.drop(columns="geometry").equals(edges_b.drop(columns="geometry"))
    assert nodes_a.drop(columns="geometry").equals(nodes_b.drop(columns="geometry"))


def test_travel_seconds():
    rules = RoadNetworkRules(speeds_kmh={"primary": 50.0, "track": 10.0})
    lengths = np.array([1000.0, 1000.0, 360.0])
    np.testing.assert_allclose(
        builder.travel_seconds(lengths, ["primary", "primary_link", "track"], rules),
        [72.0, 72.0, 129.6],
    )
    with pytest.raises(InvalidInputError, match="footway"):
        builder.travel_seconds(np.array([10.0]), ["footway"], rules)


def test_check_network_report(roads):
    report = builder.check_network(*network(roads))
    assert (report.nodes, report.edges, report.bridges, report.components) == (11, 8, 2, 3)
    assert report.largest_component_share == pytest.approx(7 / 11, abs=1e-4)
    assert report.km_by_highway["primary"] > 0


def test_check_network_rejects_fragmented_graph(roads):
    nodes, edges = network(roads)
    keep = edges["osm_id"].isin(["104", "108"])
    small = edges[keep].reset_index(drop=True)
    small["edge_id"] = range(len(small))
    used = sorted(set(small["u"]) | set(small["v"]))
    remap = {old: new for new, old in enumerate(used)}
    small["u"] = small["u"].map(remap)
    small["v"] = small["v"].map(remap)
    sub_nodes = nodes.loc[used].reset_index(drop=True)
    sub_nodes["node_id"] = range(len(sub_nodes))
    with pytest.raises(InvalidInputError, match="largest connected component"):
        builder.check_network(sub_nodes, small)


def test_check_network_rejects_bad_ids(roads):
    nodes, edges = network(roads)
    bad = edges.copy()
    bad.loc[0, "u"] = 99
    with pytest.raises(InvalidInputError, match="out of range"):
        builder.check_network(nodes, bad)
    with pytest.raises(InvalidInputError, match="edge_id"):
        builder.check_network(nodes, edges.iloc[1:])
    with pytest.raises(InvalidInputError, match="empty"):
        builder.check_network(nodes, edges.iloc[:0])


def test_clip_roads_cuts_at_boundary(roads):
    clipped = builder.clip_roads(roads, box(97.99, 4.28, 98.015, 4.32))
    assert clipped.total_bounds[2] <= 98.015 + 1e-9
    assert (clipped.geom_type == "LineString").all()


def test_region_boundary_buffers_districts():
    districts = gpd.GeoDataFrame(geometry=[box(98.0, 4.29, 98.01, 4.31)], crs="EPSG:4326")
    boundary = builder.region_boundary(districts, 1000.0, EPSG)
    assert boundary.bounds[0] < 98.0 - 0.008
    assert boundary.contains(box(98.0, 4.29, 98.01, 4.31))


def test_build_writes_outputs(tmp_path, osm, monkeypatch):
    out = tmp_path / "processed"
    out.mkdir()
    districts = gpd.GeoDataFrame(geometry=[box(98.0, 4.29, 98.02, 4.31)], crs="EPSG:4326")
    districts.to_parquet(out / "districts.parquet")
    config = tmp_path / "aceh-tamiang.yaml"
    config.write_text(json.dumps(copy.deepcopy(VALID), default=str), encoding="utf-8")
    monkeypatch.setattr(builder, "fetch", lambda remote: osm)
    report = builder.build(config, out)
    assert report.edges == 8
    nodes = gpd.read_parquet(out / "road_nodes.parquet")
    edges = gpd.read_parquet(out / "road_segments.parquet")
    graph = builder.to_road_graph(nodes, edges)
    assert (graph.n_nodes, graph.n_edges) == (10, 8)
    assert "108" not in set(edges["osm_id"])
    meta = json.loads((out / "graph.meta.json").read_text(encoding="utf-8"))
    assert meta["license"] == "ODbL 1.0"
    assert meta["utm_epsg"] == EPSG


def test_build_requires_districts(tmp_path):
    config = tmp_path / "aceh-tamiang.yaml"
    config.write_text(json.dumps(copy.deepcopy(VALID), default=str), encoding="utf-8")
    with pytest.raises(InvalidInputError, match="village boundaries first"):
        builder.build(config, tmp_path / "processed")


def test_split_drops_zero_length_segment():
    line = LineString([(98.0, 4.3), (98.0, 4.3), (98.01, 4.3)])
    assert len(builder.split_at_junctions([line])) == 1
