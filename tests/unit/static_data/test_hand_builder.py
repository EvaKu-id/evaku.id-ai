import copy
import json
import math

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin
from shapely.geometry import LineString, box

from evaku_ai.core.errors import ConfigurationError, InvalidInputError
from evaku_ai.domain.region import River
from evaku_ai.domain.rules import HandRules
from evaku_ai.geo.hand import D8_STEPS
from evaku_ai.static_data.builders import hand as builder
from tests.fixtures.region import VALID

RES = 0.0003
WEST, NORTH = 98.0, 4.03
SIZE = 100
SEA_ROW = 90
VALLEY_COL = 50


def _valley() -> np.ndarray:
    rows, cols = np.mgrid[0:SIZE, 0:SIZE]
    edge = np.minimum(np.minimum(rows, cols), SIZE - 1 - cols)
    inner = 5.0 + 0.2 * np.abs(cols - VALLEY_COL) + 0.05 * (SEA_ROW - rows)
    dem = np.where(edge < 10, 20.0 + edge, inner)
    dem[rows >= SEA_ROW] = 0.0
    return dem.astype(np.float32)


def _write_geo(path, values, west):
    profile = {
        "driver": "GTiff",
        "height": values.shape[0],
        "width": values.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(west, NORTH, RES, RES),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values, 1)
    return path


def _tiles(folder):
    folder.mkdir(parents=True, exist_ok=True)
    dem = _valley()
    half = SIZE // 2
    return [
        _write_geo(folder / "left.tif", dem[:, :half], WEST),
        _write_geo(folder / "right.tif", dem[:, half:], WEST + half * RES),
    ]


def _read(path):
    with rasterio.open(path) as src:
        return src.read(1, masked=True).astype(float).filled(np.nan), src.profile


def _write_like(path, values, profile):
    with rasterio.open(
        path, "w", **{**profile, "dtype": "float32", "nodata": builder.NODATA}
    ) as dst:
        dst.write(np.where(np.isfinite(values), values, builder.NODATA).astype(np.float32), 1)
    return path


def _pointer(dem):
    pointer = np.full(dem.shape, np.nan)
    for r, c in zip(*np.nonzero(np.isfinite(dem)), strict=True):
        best, code = 0.0, 0
        for step, (dr, dc) in D8_STEPS.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < dem.shape[0] and 0 <= nc < dem.shape[1] and np.isfinite(dem[nr, nc]):
                slope = (dem[r, c] - dem[nr, nc]) / math.hypot(dr, dc)
                if slope > best:
                    best, code = slope, step
        pointer[r, c] = code
    return pointer


def _downstream(pointer, r, c):
    code = int(pointer[r, c])
    if code == 0:
        return None
    dr, dc = D8_STEPS[code]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < pointer.shape[0] and 0 <= nc < pointer.shape[1]):
        return None
    return (nr, nc) if np.isfinite(pointer[nr, nc]) else None


def _path(pointer, r, c):
    cell = (r, c)
    while cell is not None:
        yield cell
        cell = _downstream(pointer, *cell)


class FakeWhitebox:
    def breach(self, binary, dem, output, dist):
        values, profile = _read(dem)
        return _write_like(output, values, profile)

    def d8_pointer(self, binary, dem, output):
        values, profile = _read(dem)
        return _write_like(output, _pointer(values), profile)

    def flow_accumulation(self, binary, pointer_path, output):
        pointer, profile = _read(pointer_path)
        acc = np.where(np.isfinite(pointer), 0.0, np.nan)
        for r, c in zip(*np.nonzero(np.isfinite(pointer)), strict=True):
            for cell in _path(pointer, r, c):
                acc[cell] += 1
        return _write_like(output, acc, profile)

    def extract_streams(self, binary, acc_path, output, threshold):
        acc, profile = _read(acc_path)
        return _write_like(
            output, np.where(np.isfinite(acc), (acc >= threshold) * 1.0, np.nan), profile
        )

    def elevation_above_stream(self, binary, dem_path, streams_path, output):
        dem, profile = _read(dem_path)
        streams = np.nan_to_num(_read(streams_path)[0]) > 0
        pointer = _pointer(dem)
        hand = np.full(dem.shape, np.nan)
        for r, c in zip(*np.nonzero(np.isfinite(dem)), strict=True):
            for cell in _path(pointer, r, c):
                if streams[cell]:
                    hand[r, c] = dem[r, c] - dem[cell]
                    break
        return _write_like(output, hand, profile)

    def watershed(self, binary, pointer_path, pour_points, output):
        pointer, profile = _read(pointer_path)
        point = gpd.read_file(pour_points).geometry.iloc[0]
        outlet = rasterio.transform.rowcol(profile["transform"], point.x, point.y)
        basin = np.full(pointer.shape, np.nan)
        for r, c in zip(*np.nonzero(np.isfinite(pointer)), strict=True):
            if any(cell == tuple(outlet) for cell in _path(pointer, r, c)):
                basin[r, c] = 1.0
        return _write_like(output, basin, profile)

    def streams_to_vector(self, binary, streams_path, pointer_path, output):
        streams, profile = _read(streams_path)
        pointer, _ = _read(pointer_path)
        transform = profile["transform"]
        lines = []
        for r, c in zip(*np.nonzero(np.nan_to_num(streams) > 0), strict=True):
            nxt = _downstream(pointer, r, c)
            if nxt is not None:
                a = transform @ (c + 0.5, r + 0.5)
                b = transform @ (nxt[1] + 0.5, nxt[0] + 0.5)
                lines.append(LineString([a, b]))
        gpd.GeoDataFrame(geometry=lines, crs=profile["crs"]).to_file(output)
        return output


def _region_file(tmp_path, tiles):
    region = copy.deepcopy(VALID)
    region["bbox"] = [98.006, 4.006, 98.024, 4.026]
    region["dem_bbox"] = [WEST, NORTH - SIZE * RES, WEST + SIZE * RES, NORTH]
    region["rivers"][0]["glofas_point"] = [98.015, 4.012]
    region["inputs"]["dem"] = [
        {**VALID["inputs"]["dem"][0], "path": str(path.relative_to(tmp_path))} for path in tiles
    ]
    region["hand"] = {"stream_threshold_km2": 0.05, "pour_point_snap_m": 200}
    path = tmp_path / "aceh-tamiang.yaml"
    path.write_text(yaml.safe_dump(region, sort_keys=False), encoding="utf-8")
    return path


def _patch_io(monkeypatch, fake):
    monkeypatch.setattr(builder, "fetch", lambda remote: remote.path)
    monkeypatch.setattr(builder.whitebox, "install", lambda package, dest: dest / "whitebox_tools")
    monkeypatch.setattr(builder.whitebox, "version", lambda binary: "WhiteboxTools fake")
    for name in (
        "breach",
        "d8_pointer",
        "flow_accumulation",
        "extract_streams",
        "elevation_above_stream",
        "watershed",
        "streams_to_vector",
    ):
        monkeypatch.setattr(builder.whitebox, name, getattr(fake, name))
    river = LineString([(98.015, 4.029), (98.015, 4.004)])
    monkeypatch.setattr(
        builder,
        "osm_waterways",
        lambda path, bbox: gpd.GeoDataFrame(
            {"osm_id": ["1"], "waterway": ["river"]}, geometry=[river], crs="EPSG:4326"
        ),
    )


def test_build_writes_hand_streams_basins_and_meta(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tiles = _tiles(tmp_path / "data" / "raw")
    out = tmp_path / "data" / "processed"
    out.mkdir(parents=True)
    gpd.GeoDataFrame(
        {"district_id": ["1"]}, geometry=[box(98.006, 4.006, 98.024, 4.026)], crs="EPSG:4326"
    ).to_parquet(out / "districts.parquet")
    _patch_io(monkeypatch, FakeWhitebox())

    report = builder.build(_region_file(tmp_path, tiles), out)

    assert report.stream_threshold_cells == 56
    assert 0 < report.regency_share_in_basin["tamiang"] <= 1
    assert report.osm_river_recall > 0.9
    assert report.burned_river_km > 0
    assert report.burn_clipped_cells >= 0
    values = list(report.median_by_band_m.values())
    assert values == sorted(values)
    with rasterio.open(out / "hand.tif") as src:
        assert src.crs.to_epsg() == 32647
        assert src.nodata == builder.NODATA
        assert src.profile["compress"].lower() == "deflate"
        assert src.overviews(1) is not None
    basins = gpd.read_parquet(out / "river_basins.parquet")
    assert basins["river_id"].tolist() == ["tamiang"]
    assert gpd.read_parquet(out / "streams.parquet").crs.to_epsg() == 4326
    meta = json.loads((out / "hand.meta.json").read_text())
    assert meta["whitebox"]["version"] == "WhiteboxTools fake"
    assert meta["crs"] == "EPSG:32647"
    assert not list((tmp_path / "data" / "interim").iterdir())


def test_build_requires_districts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tiles = _tiles(tmp_path / "data" / "raw")
    with pytest.raises(InvalidInputError, match="districts"):
        builder.build(_region_file(tmp_path, tiles), tmp_path / "data" / "processed")


def test_prepare_dem_reprojects(tmp_path):
    tiles = _tiles(tmp_path)
    dem_bbox = (WEST, NORTH - SIZE * RES, WEST + SIZE * RES, NORTH)
    path = builder.prepare_dem(tiles, dem_bbox, 32647, 30.0, tmp_path / "dem.tif")
    values, profile = _read(path)
    assert profile["crs"].to_epsg() == 32647
    assert profile["transform"].a == pytest.approx(30.0)
    assert np.nanmin(values) == 0.0


def test_prepare_dem_rejects_uncovered_bbox(tmp_path):
    tiles = _tiles(tmp_path)
    with pytest.raises(ConfigurationError, match="do not cover"):
        builder.prepare_dem(tiles, (WEST, 3.9, 98.03, NORTH), 32647, 30.0, tmp_path / "dem.tif")


def test_locate_outlet_snaps_then_follows_flow():
    pointer = np.array([[0, 8, 0], [0, 8, 0], [0, 8, 0]])
    acc = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 3.0, 1.0]])
    land = np.array([[True] * 3, [True] * 3, [True, True, True]])
    transform = from_origin(0.0, 90.0, 30.0, 30.0)
    assert builder.locate_outlet(pointer, acc, land, transform, (5.0, 85.0), 1) == (2, 1)


def _check_inputs():
    hand = np.tile(np.abs(np.arange(41) - 20) * 2.0, (5, 1))
    streams = np.zeros(hand.shape, dtype=bool)
    streams[:, 20] = True
    land = np.ones(hand.shape, dtype=bool)
    return hand, streams, land


def test_check_hand_reports_medians_and_percentiles():
    hand, streams, land = _check_inputs()
    rules = HandRules(distance_bands_m=(100.0, 300.0, 600.0))
    medians, (p5, p50, p95) = builder.check_hand(hand, streams, land, land, rules)
    assert medians.tolist() == sorted(medians.tolist())
    assert p5 <= p50 <= p95


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda h, s, land: h.__setitem__((0, 20), 1.0), "stream cells must be 0"),
        (lambda h, s, land: h.__setitem__((0, 0), -1.0), "negative HAND"),
        (lambda h, s, land: h.__setitem__((0, 0), np.nan), "no data on land"),
        (lambda h, s, land: s.__setitem__(slice(None), False), "no stream cells"),
        (lambda h, s, land: h.__setitem__(slice(None), np.where(s, 0.0, 5.0)), "must rise"),
    ],
)
def test_check_hand_rejects_bad_rasters(change, message):
    hand, streams, land = _check_inputs()
    change(hand, streams, land)
    rules = HandRules(distance_bands_m=(100.0, 300.0, 600.0))
    with pytest.raises(InvalidInputError, match=message):
        builder.check_hand(hand, streams, land, land, rules)


def test_basin_touching_the_edge_is_rejected(tmp_path, monkeypatch):
    fake = FakeWhitebox()

    def everything(binary, pointer_path, pour_points, output):
        pointer, profile = _read(pointer_path)
        return _write_like(output, np.ones(pointer.shape), profile)

    monkeypatch.setattr(builder.whitebox, "watershed", everything)
    values, profile = _utm_valley()
    dem = _write_like(tmp_path / "dem.tif", values, profile)
    pointer_path = fake.d8_pointer(None, dem, tmp_path / "d8.tif")
    acc = _read(fake.flow_accumulation(None, pointer_path, tmp_path / "acc.tif"))[0]
    transform = profile["transform"]
    pointer = np.nan_to_num(_read(pointer_path)[0]).astype(np.int64)
    flow = builder.Flow(pointer_path, pointer, acc, np.isfinite(acc), transform, 32647)
    x, y = transform @ (VALLEY_COL + 0.5, SEA_ROW // 2)
    centre = gpd.GeoSeries.from_xy([x], [y], crs="EPSG:32647").to_crs("EPSG:4326").iloc[0]
    river = River(river_id="tamiang", name="Sungai Tamiang", glofas_point=(centre.x, centre.y))
    with pytest.raises(ConfigurationError, match="touches the DEM edge"):
        builder._basins([river], flow, HandRules(), tmp_path / "wbt", tmp_path)


def _utm_valley():
    values = _valley()[:SEA_ROW].astype(float)
    profile = {
        "driver": "GTiff",
        "height": values.shape[0],
        "width": values.shape[1],
        "count": 1,
        "crs": "EPSG:32647",
        "transform": from_origin(500000.0, 446000.0, 30.0, 30.0),
    }
    return values, profile


def test_condition_dem_burns_rivers_and_keeps_estuaries(tmp_path):
    values = _valley().astype(float)
    values[SEA_ROW - 5 : SEA_ROW, VALLEY_COL] = 0.0
    _, profile = _utm_valley()
    profile = {**profile, "height": SIZE}
    dem = _write_like(tmp_path / "dem.tif", values, profile)
    x0, y0 = profile["transform"] @ (VALLEY_COL + 0.5, 0)
    x1, y1 = profile["transform"] @ (VALLEY_COL + 0.5, SEA_ROW - 0.5)
    rivers = gpd.GeoDataFrame(geometry=[LineString([(x0, y0), (x1, y1)])], crs="EPSG:32647")
    out, river_cells = builder.condition_dem(dem, rivers, 10.0, tmp_path / "burned.tif")
    burned_m = np.where(river_cells, 10.0, 0.0)
    burned, _ = _read(out)
    assert np.isnan(burned[SEA_ROW:]).all()
    assert burned[SEA_ROW - 1, VALLEY_COL] == -10.0
    assert burned_m[:SEA_ROW, VALLEY_COL].min() == 10.0
    assert burned_m[:, 10].max() == 0.0
    land = np.isfinite(burned)
    np.testing.assert_allclose((burned + burned_m)[land], values[land], atol=1e-4)
    plain, none = builder.condition_dem(dem, rivers.iloc[:0], 10.0, tmp_path / "plain.tif")
    assert not none.any()
    assert np.isnan(_read(plain)[0][SEA_ROW - 1, VALLEY_COL])
    unburned, kept = builder.condition_dem(dem, rivers, 0.0, tmp_path / "zero.tif")
    assert kept[SEA_ROW - 1, VALLEY_COL]
    assert _read(unburned)[0][SEA_ROW - 1, VALLEY_COL] == 0.0


def test_add_mapped_rivers_marks_river_cells_as_streams(tmp_path):
    values, profile = _utm_valley()
    extracted = np.where(np.arange(values.shape[1]) == VALLEY_COL, 1.0, 0.0) * np.ones_like(values)
    extracted[0, 0] = np.nan
    path = _write_like(tmp_path / "extracted.tif", extracted, profile)
    rivers = np.zeros(values.shape, dtype=bool)
    rivers[:, 10] = True
    merged, _ = _read(builder.add_mapped_rivers(path, rivers, tmp_path / "streams.tif"))
    assert (merged[1:, VALLEY_COL] == 1).all() and (merged[1:, 10] == 1).all()
    assert merged[5, 20] == 0
    assert np.isnan(merged[0, 0])
