import copy
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, box

from evaku_ai.core.errors import InvalidInputError
from evaku_ai.engine.accessibility import NodePopulation
from evaku_ai.static_data.builders import population as builder
from tests.fixtures.region import VALID

EPSG = 32647
FINE = 0.0025
VILLAGES = gpd.GeoDataFrame(
    {"village_id": ["1114010001", "1114010002", "1114010003"]},
    geometry=[
        box(98.00, 4.20, 98.01, 4.21),
        box(98.01, 4.20, 98.02, 4.21),
        box(98.02, 4.20, 98.03, 4.21),
    ],
    crs="EPSG:4326",
)


def write_tif(path, values, west, north, res):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(west, north, res, res),
        nodata=-99999,
    ) as dst:
        dst.write(values.astype("float32"), 1)
    return path


def fine_raster(tmp_path):
    values = np.ones((4, 16))
    values[:, 14:] = 5.0
    return write_tif(tmp_path / "fine.tif", values, 98.00, 4.21, FINE)


def coarse_rasters(tmp_path):
    total = np.array([[100.0, 100.0, 0.0]])
    under5 = np.array([[10.0, 10.0, 0.0]])
    elderly = np.array([[20.0, 30.0, 0.0]])
    return (
        write_tif(tmp_path / "total.tif", total, 97.995, 4.22, 0.02),
        write_tif(tmp_path / "u5.tif", under5, 97.995, 4.22, 0.02),
        write_tif(tmp_path / "eld.tif", elderly, 97.995, 4.22, 0.02),
    )


def graph():
    points = [
        Point(98.002, 4.205),
        Point(98.0102, 4.205),
        Point(98.015, 4.205),
        Point(98.2, 4.3),
        Point(98.201, 4.3),
    ]
    nodes = gpd.GeoDataFrame({"node_id": range(5)}, geometry=points, crs="EPSG:4326")
    projected = nodes.to_crs(epsg=EPSG)
    nodes["x_m"], nodes["y_m"] = projected.geometry.x, projected.geometry.y
    edges = gpd.GeoDataFrame(
        {
            "edge_id": [0, 1, 2],
            "u": [0, 1, 3],
            "v": [1, 2, 4],
            "highway_type": ["primary"] * 3,
            "is_bridge": [False] * 3,
            "length_m": [100.0] * 3,
            "travel_s": [7.2] * 3,
        },
        geometry=[LineString([(0, 0), (1, 1)])] * 3,
        crs="EPSG:4326",
    )
    return nodes, edges


@pytest.fixture
def pixels(tmp_path):
    grid = builder.read_window(fine_raster(tmp_path), (98.0, 4.2, 98.04, 4.21))
    return builder.pixels_in_villages(grid, VILLAGES)


def test_pixels_outside_villages_are_dropped(pixels):
    assert len(pixels) == 4 * 12
    assert pixels["population"].sum() == pytest.approx(48.0)
    assert set(pixels["village_id"]) == set(VILLAGES["village_id"])


def test_age_shares_and_fallback(tmp_path, pixels):
    total, under5, elderly = coarse_rasters(tmp_path)
    window = (97.99, 4.19, 98.06, 4.23)
    shares = builder.age_shares(
        pixels,
        builder.read_window(total, window),
        {
            "under5": [builder.read_window(under5, window)],
            "elderly": [builder.read_window(elderly, window)],
        },
    )
    np.testing.assert_allclose(shares["under5"], 0.1)
    assert set(np.round(shares["elderly"], 3)) <= {0.2, 0.3}


def test_age_share_without_coarse_data_uses_regency_average(tmp_path, pixels):
    empty = builder.Grid(np.zeros((1, 1)), (1.0, 0.0, 0.0, 0.0, -1.0, 0.0), None)
    shares = builder.age_shares(pixels, empty, {"under5": [empty]})
    np.testing.assert_array_equal(shares["under5"], 0.0)


def test_snap_stays_inside_village(pixels):
    nodes, _ = graph()
    villages_of_nodes = builder.node_villages(nodes, VILLAGES)
    assert list(villages_of_nodes[:3]) == ["1114010001", "1114010002", "1114010002"]
    node_id, without = builder.snap_by_village(pixels, nodes, villages_of_nodes, EPSG)
    in_a = (pixels["village_id"] == "1114010001").to_numpy()
    assert set(node_id[in_a]) == {0}
    assert without == ["1114010003"]
    assert set(node_id[(pixels["village_id"] == "1114010003").to_numpy()]) <= {2}


def tables(pixels, villages=VILLAGES):
    nodes, _ = graph()
    villages_of_nodes = builder.node_villages(nodes, villages)
    node_id, without = builder.snap_by_village(pixels, nodes, villages_of_nodes, EPSG)
    shares = {"under5": np.full(len(pixels), 0.1), "elderly": np.full(len(pixels), 0.2)}
    node_table, village_table = builder.aggregate(
        pixels, node_id, shares, nodes, villages_of_nodes, list(villages["village_id"])
    )
    return node_table, village_table, without


def test_aggregate_conserves_people(pixels):
    node_table, village_table, _ = tables(pixels)
    assert node_table["population"].sum() == pytest.approx(48.0)
    assert village_table["population"].tolist() == pytest.approx([16.0, 16.0, 16.0])
    assert village_table["children_under5"].sum() == pytest.approx(4.8)
    population = NodePopulation(
        village=np.array([0, 1, 1, -1, -1]),
        total=node_table["population"].to_numpy(),
        under5=node_table["under5"].to_numpy(),
        elderly=node_table["elderly"].to_numpy(),
    )
    assert population.total.sum() == pytest.approx(48.0)


def test_check_population(pixels):
    node_table, village_table, without = tables(pixels)
    report = builder.check_population(48.0, node_table, village_table, without, 0.0)
    assert (report.villages, report.under5_share, report.elderly_share) == (3, 0.1, 0.2)
    assert report.villages_without_nodes == ["1114010003"]
    with pytest.raises(InvalidInputError, match="raster total"):
        builder.check_population(50.0, node_table, village_table, without, 0.0)
    empty = village_table.copy()
    empty.loc[2, "population"] = 0.0
    empty.loc[2, "children_under5"] = 0.0
    empty.loc[2, "elderly"] = 0.0
    with pytest.raises(InvalidInputError, match="1114010003"):
        builder.check_population(
            32.0, node_table.assign(population=node_table["population"] * 32 / 48), empty, [], 0.0
        )
    too_many = village_table.assign(elderly=village_table["population"] * 2)
    with pytest.raises(InvalidInputError, match="elderly"):
        builder.check_population(48.0, node_table, too_many, without, 0.0)


def test_build_end_to_end(tmp_path, monkeypatch):
    out = tmp_path / "processed"
    out.mkdir()
    VILLAGES.to_parquet(out / "villages.parquet")
    nodes, edges = graph()
    nodes.to_parquet(out / "road_nodes.parquet")
    edges.to_parquet(out / "road_segments.parquet")
    fine = fine_raster(tmp_path)
    total, under5, elderly = coarse_rasters(tmp_path)
    by_name = {"pop_100m.tif": fine, "pop_1km.tif": total, "t_00.tif": under5, "t_60.tif": elderly}
    monkeypatch.setattr(builder, "fetch", lambda remote: by_name[remote.path.name])
    config = tmp_path / "aceh-tamiang.yaml"
    config.write_text(json.dumps(copy.deepcopy(VALID), default=str), encoding="utf-8")

    report = builder.build(config, out)
    assert report.total == pytest.approx(48.0)
    node_table = pd.read_parquet(out / "node_population.parquet")
    assert node_table["population"].sum() == pytest.approx(48.0)
    assert len(pd.read_parquet(out / "village_population.parquet")) == 3
    meta = json.loads((out / "population.meta.json").read_text(encoding="utf-8"))
    assert meta["license"] == "CC BY 4.0"
    assert report.outside_main_component_share == pytest.approx(0.0)


def test_build_requires_static_layers(tmp_path):
    config = tmp_path / "aceh-tamiang.yaml"
    config.write_text(json.dumps(copy.deepcopy(VALID), default=str), encoding="utf-8")
    with pytest.raises(InvalidInputError, match="road graph first"):
        builder.build(config, tmp_path / "processed")
