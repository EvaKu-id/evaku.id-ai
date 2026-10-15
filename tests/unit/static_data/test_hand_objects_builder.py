import copy
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, box

from evaku_ai.core.errors import InvalidInputError
from evaku_ai.domain.rules import HandRules
from evaku_ai.static_data.builders import hand_objects as builder
from tests.fixtures.region import VALID

UTM = "EPSG:32647"
X0, Y0, CELL, SIZE, STREAM_COL, SEA_COL = 400000.0, 460000.0, 30.0, 40, 20, 36


def _x(col):
    return X0 + (col + 0.5) * CELL


def _y(row):
    return Y0 - (row + 0.5) * CELL


def _hand_values():
    values = np.tile(np.abs(np.arange(SIZE) - STREAM_COL) * 1.0, (SIZE, 1))
    values[:, SEA_COL:] = np.nan
    return values


def _grid(footprint=None):
    return builder.HandGrid(
        _hand_values(),
        (CELL, 0.0, X0, 0.0, -CELL, Y0),
        footprint or box(X0, Y0 - SIZE * CELL, X0 + SIZE * CELL, Y0),
    )


def _stream():
    return LineString([(_x(STREAM_COL), Y0), (_x(STREAM_COL), Y0 - SIZE * CELL)])


def test_sample_hand_sea_is_zero_and_outside_is_nan():
    x = np.array([_x(STREAM_COL + 3), _x(SEA_COL + 1), X0 - 500.0])
    y = np.full(3, _y(5))
    result = builder.sample_hand(_grid(), x, y)
    assert result[:2].tolist() == [3.0, 0.0]
    assert np.isnan(result[2])


def test_edge_hand_skips_crossings_but_keeps_riverbank_roads():
    parallel = LineString([(_x(STREAM_COL + 2), _y(2)), (_x(STREAM_COL + 2), _y(30))])
    crossing = LineString([(_x(10), _y(10)), (_x(30), _y(10))])
    short = LineString([(_x(STREAM_COL) - 20, _y(15)), (_x(STREAM_COL) + 20, _y(15))])
    bank = LineString([(_x(STREAM_COL) + 5, _y(20)), (_x(STREAM_COL) + 5, _y(25))])
    outside = LineString([(X0 - 900, _y(5)), (X0 - 600, _y(5))])
    lines = [parallel, crossing, short, bank, outside]
    hand, crossings = builder.edge_hand(_grid(), lines, [_stream()], HandRules())
    assert hand[0] == 2.0
    assert hand[1] == 2.0
    assert hand[2] == 0.0
    assert hand[3] == 0.0
    assert np.isnan(hand[4])
    assert crossings.tolist() == [0, 1, 1, 0, 0]


def test_node_hand_uses_population_weighted_median_with_fallback():
    nodes = pd.DataFrame(
        {"node_id": [0, 1], "x_m": [_x(STREAM_COL + 4), _x(STREAM_COL + 6)], "y_m": [_y(5)] * 2}
    )
    pixels = pd.DataFrame(
        {"node_id": [0, 0, 0], "hand": [1.0, 9.0, 2.0], "population": [5.0, 1.0, 1.0]}
    )
    table = builder.node_hand(_grid(), nodes, pixels)
    assert table["hand"].tolist() == [1.0, 6.0]
    assert table["hand_at_node"].tolist() == [4.0, 6.0]


def test_village_hand_median_and_low_population():
    pixels = pd.DataFrame(
        {
            "village_id": ["a", "a", "a", "b"],
            "hand": [0.5, 2.0, 8.0, 5.0],
            "population": [10.0, 5.0, 5.0, 3.0],
        }
    )
    table = builder.village_hand(pixels, ["a", "b", "c"], HandRules())
    assert table["village_id"].tolist() == ["a", "b", "c"]
    assert table["hand_median"].tolist()[:2] == [0.5, 5.0]
    assert np.isnan(table["hand_median"].iloc[2])
    assert table["population_hand_le_1m"].tolist() == [10.0, 0.0, 0.0]
    assert table["population_hand_le_3m"].tolist() == [15.0, 0.0, 0.0]


def test_assign_river_inside_and_outside_basin():
    basins = gpd.GeoDataFrame({"river_id": ["tamiang"]}, geometry=[box(0, 0, 10, 10)], crs=UTM)
    points = gpd.GeoSeries([Point(5, 5), Point(20, 20)], crs=UTM)
    result = builder.assign_river(points, basins)
    assert result.iloc[0] == "tamiang"
    assert pd.isna(result.iloc[1])


def _tables(edge_hand=(1.0, 6.0), facility_hand=(2.0,), node_hand=(1.0,)):
    edges = pd.DataFrame({"hand_min": list(edge_hand)})
    nodes = pd.DataFrame({"hand": list(node_hand)})
    facilities = pd.DataFrame({"hand": list(facility_hand)})
    return edges, nodes, facilities


def test_check_objects_reports_near_and_far_medians():
    edges, nodes, facilities = _tables()
    near, far = builder.check_objects(
        edges, np.array([True, True]), nodes, facilities, np.array([50.0, 900.0]), HandRules()
    )
    assert (near, far) == (1.0, 6.0)


@pytest.mark.parametrize(
    ("tables", "inside", "message"),
    [
        (_tables(edge_hand=(np.nan, 6.0)), [True, True], "have no HAND"),
        (_tables(facility_hand=(np.nan,)), [True, True], "every facility"),
        (_tables(node_hand=(-1.0,)), [True, True], "negative node hand"),
        (_tables(edge_hand=(7.0, 6.0)), [True, True], "near streams should sit lower"),
    ],
)
def test_check_objects_rejects_bad_tables(tables, inside, message):
    edges, nodes, facilities = tables
    with pytest.raises(InvalidInputError, match=message):
        builder.check_objects(
            edges, np.array(inside), nodes, facilities, np.array([50.0, 900.0]), HandRules()
        )


def test_check_objects_allows_missing_hand_outside_the_dem():
    edges, nodes, facilities = _tables(edge_hand=(1.0, 6.0, np.nan))
    distance = np.array([50.0, 900.0, 900.0])
    builder.check_objects(
        edges, np.array([True, True, False]), nodes, facilities, distance, HandRules()
    )


def _to_wgs(geoms):
    return gpd.GeoSeries(geoms, crs=UTM).to_crs("EPSG:4326")


def _write_raster(path, values, transform, crs):
    profile = {
        "driver": "GTiff",
        "height": values.shape[0],
        "width": values.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": -9999.0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.where(np.isfinite(values), values, -9999.0).astype(np.float32), 1)


def _inputs(tmp_path):
    out = tmp_path / "data" / "processed"
    out.mkdir(parents=True)
    _write_raster(out / "hand.tif", _hand_values(), from_origin(X0, Y0, CELL, CELL), UTM)
    (out / "hand.meta.json").write_text(json.dumps({"built_at": "2026-10-15T00:00:00+00:00"}))
    extent = box(X0, Y0 - SIZE * CELL, X0 + SIZE * CELL, Y0)
    gpd.GeoDataFrame(
        {"river_id": ["tamiang"]}, geometry=_to_wgs([box(X0, Y0 - 600, X0 + 900, Y0)])
    ).to_parquet(out / "river_basins.parquet")
    gpd.GeoDataFrame(geometry=_to_wgs([_stream()])).to_parquet(out / "streams.parquet")
    node_xy = [
        (_x(STREAM_COL + 1), _y(5)),
        (_x(STREAM_COL + 1), _y(30)),
        (_x(2), _y(5)),
        (_x(2), _y(30)),
    ]
    gpd.GeoDataFrame(
        {"node_id": range(4), "x_m": [p[0] for p in node_xy], "y_m": [p[1] for p in node_xy]},
        geometry=_to_wgs([Point(p) for p in node_xy]),
    ).to_parquet(out / "road_nodes.parquet")
    roads = [
        LineString([node_xy[0], node_xy[1]]),
        LineString([node_xy[2], node_xy[3]]),
        LineString([node_xy[2], node_xy[0]]),
    ]
    gpd.GeoDataFrame(
        {"edge_id": range(3), "u": [0, 2, 2], "v": [1, 3, 0]}, geometry=_to_wgs(roads)
    ).to_parquet(out / "road_segments.parquet")
    gpd.GeoDataFrame(
        {"facility_id": [0, 1], "name": ["Puskesmas", ""], "type": ["health_center", "school"]},
        geometry=_to_wgs([Point(_x(STREAM_COL + 3), _y(8)), Point(_x(4), _y(25))]),
    ).to_parquet(out / "facilities.parquet")
    halves = [
        box(X0, Y0 - SIZE * CELL, X0 + SIZE * CELL / 2, Y0),
        box(X0 + SIZE * CELL / 2, Y0 - SIZE * CELL, X0 + SIZE * CELL, Y0),
    ]
    gpd.GeoDataFrame(
        {"village_id": ["1114000001", "1114000002"]}, geometry=_to_wgs(halves)
    ).to_parquet(out / "villages.parquet")

    lon_lat = _to_wgs([extent]).total_bounds
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    step = (lon_lat[2] - lon_lat[0]) / 12
    _write_raster(
        raw / "pop.tif",
        np.full((12, 12), 10.0),
        from_origin(lon_lat[0], lon_lat[3], step, (lon_lat[3] - lon_lat[1]) / 12),
        "EPSG:4326",
    )
    region = copy.deepcopy(VALID)
    region["bbox"] = [float(v) for v in lon_lat]
    region["dem_bbox"] = [
        float(lon_lat[0] - 0.01),
        float(lon_lat[1] - 0.01),
        float(lon_lat[2] + 0.01),
        float(lon_lat[3] + 0.01),
    ]
    region["rivers"][0]["glofas_point"] = [float(lon_lat[0]), float(lon_lat[1])]
    region["inputs"]["population"]["total_100m"]["path"] = "data/raw/pop.tif"
    path = tmp_path / "aceh-tamiang.yaml"
    path.write_text(yaml.safe_dump(region, sort_keys=False), encoding="utf-8")
    return path, out


def test_build_writes_tables_and_meta(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    region, out = _inputs(tmp_path)
    monkeypatch.setattr(builder, "fetch", lambda remote: remote.path)

    report = builder.build(region, out)

    edges = pd.read_parquet(out / "edge_hand.parquet")
    assert edges["edge_id"].tolist() == [0, 1, 2]
    assert edges["hand_min"].tolist()[:2] == [1.0, 18.0]
    assert edges["crossings"].tolist() == [0, 0, 1]
    assert edges["river_id"].tolist()[0] == "tamiang"
    nodes = pd.read_parquet(out / "node_hand.parquet")
    assert len(nodes) == 4 and np.isfinite(nodes["hand"]).all()
    facilities = pd.read_parquet(out / "facility_hand.parquet")
    assert facilities["hand"].tolist() == [3.0, 16.0]
    villages = pd.read_parquet(out / "village_hand.parquet")
    assert villages["village_id"].tolist() == ["1114000001", "1114000002"]
    assert report.edges == 3 and report.edges_with_skipped_crossings == 1
    assert report.lowest_facilities[1] == "school #1 (16.0 m)"
    meta = json.loads((out / "hand_objects.meta.json").read_text())
    assert meta["rules"]["crossing_buffer_m"] == 45.0
    assert len(meta["hand_sha256"]) == 64


def test_build_requires_inputs(tmp_path):
    region, out = _inputs(tmp_path)
    (out / "streams.parquet").unlink()
    with pytest.raises(InvalidInputError, match=r"streams\.parquet"):
        builder.build(region, out)
