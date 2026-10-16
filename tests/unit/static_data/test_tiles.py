import json
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import pytest
from shapely.geometry import LineString, box

from evaku_ai.adapters.download import file_sha256
from evaku_ai.adapters.tiles_release import parse_tiles_manifest
from evaku_ai.core.errors import DataNotLoadedError, InvalidInputError
from evaku_ai.static_data.builders import tiles
from evaku_ai.static_data.builders.tiles import ROADS, VILLAGES, TileLayer, tippecanoe_args

VERSION = "2026-10-16.3"


def test_roads_args_keep_only_listed_attributes():
    args = tippecanoe_args(ROADS, Path("in/roads.geojson"), Path("out"))
    assert args[0] == "tippecanoe"
    assert args[args.index("--output") + 1] == str(Path("out/jalan.pmtiles"))
    assert args[args.index("--layer") + 1] == "ruas"
    assert [a for a in args if a.startswith("--include=")] == [
        "--include=edge_id",
        "--include=highway_type",
        "--include=is_bridge",
    ]
    assert "--quiet" in args and "--description=EvaKu ruas" in args
    assert "--attribution=© OpenStreetMap contributors" in args
    assert not any(a.startswith("--use-attribute-for-id") for a in args)
    assert args[-1] == str(Path("in/roads.geojson"))


def test_villages_args_carry_the_boundary_attribution():
    args = tippecanoe_args(VILLAGES, Path("v.geojson"), Path("out"), binary=Path("bin/tc"))
    assert args[0] == str(Path("bin/tc"))
    assert args[args.index("--maximum-zoom") + 1] == "13"
    assert any(a.startswith("--attribution=Batas desa") for a in args)


def test_layer_without_attribution_has_no_flag():
    layer = TileLayer("x", "x.pmtiles", ("id",), 0, 10)
    assert not any(
        a.startswith("--attribution") for a in tippecanoe_args(layer, Path("s"), Path("o"))
    )


def test_args_are_a_list_without_shell_quoting():
    args = tippecanoe_args(VILLAGES, Path("dir with space/v.geojson"), Path("out"))
    assert args[-1] == str(Path("dir with space/v.geojson"))


@pytest.mark.parametrize(("min_zoom", "max_zoom"), [(-1, 10), (12, 10), (5, 23)])
def test_layer_rejects_invalid_zoom(min_zoom, max_zoom):
    with pytest.raises(ValueError, match="zoom"):
        TileLayer("x", "x.pmtiles", ("id",), min_zoom, max_zoom)


def test_layer_rejects_id_outside_attributes():
    with pytest.raises(ValueError, match="attributes"):
        TileLayer("x", "x.pmtiles", ("name",), 0, 10, integer_id="id")


def _processed(path: Path, utm_roads: bool = False) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    roads = gpd.GeoDataFrame(
        {
            "edge_id": [0, 1],
            "u": [0, 1],
            "v": [1, 2],
            "highway_type": ["primary", "track"],
            "is_bridge": [False, True],
            "osm_id": ["w1", "w2"],
        },
        geometry=[LineString([(98.0, 4.0), (98.1, 4.0)]), LineString([(98.1, 4.0), (98.2, 4.1)])],
        crs=4326,
    )
    if utm_roads:
        roads = roads.to_crs(32647)
    roads.to_parquet(path / tiles.ROADS_SOURCE)
    gpd.GeoDataFrame(
        {"village_id": ["1114000001"], "name": ["Alpha"], "district_id": ["1114000"]},
        geometry=[box(98.0, 4.0, 98.1, 4.1)],
        crs=4326,
    ).to_parquet(path / tiles.VILLAGES_SOURCE)
    return path


class FakeTippecanoe:
    def __init__(self, size: int = 64) -> None:
        self.size = size
        self.inputs: dict[str, list[dict[str, object]]] = {}
        self.args: list[list[str]] = []

    def run(self, args: list[str], output: Path) -> Path:
        self.args.append(args)
        source = Path(args[-1])
        lines = source.read_text(encoding="utf-8").splitlines()
        self.inputs[output.name] = [json.loads(line) for line in lines]
        output.write_bytes(output.name.encode() * self.size)
        return output


@pytest.fixture
def fake(monkeypatch, tmp_path):
    tool = FakeTippecanoe()
    monkeypatch.setattr(tiles.tippecanoe, "find_binary", lambda: tmp_path / "tippecanoe")
    monkeypatch.setattr(tiles.tippecanoe, "version", lambda binary: "tippecanoe v2.79.0")
    monkeypatch.setattr(tiles.tippecanoe, "run", tool.run)
    return tool


def _bundle_of(processed: Path, **overrides):
    sources = {name: file_sha256(processed / name) for name in tiles.SOURCES.values()}
    manifest = SimpleNamespace(sources=sources | overrides)

    def read(root, version):
        return SimpleNamespace(manifest=manifest)

    return read


def _build(monkeypatch, tmp_path, processed=None, **overrides):
    processed = processed or _processed(tmp_path / "processed")
    monkeypatch.setattr(tiles, "read_bundle", _bundle_of(processed, **overrides))
    return tiles.build(VERSION, processed, tmp_path / "static", tmp_path / "tiles")


def test_build_writes_both_layers_and_a_manifest(monkeypatch, tmp_path, fake):
    manifest = _build(monkeypatch, tmp_path)
    out = tmp_path / "tiles" / VERSION
    assert sorted(p.name for p in out.iterdir()) == ["desa.pmtiles", "jalan.pmtiles", "tiles.json"]
    assert sorted(p.name for p in (tmp_path / "tiles").iterdir()) == [VERSION]
    assert parse_tiles_manifest((out / "tiles.json").read_bytes()) == manifest
    assert manifest.tippecanoe == "tippecanoe v2.79.0"
    roads = {f.path: f for f in manifest.files}["jalan.pmtiles"]
    assert (roads.layer, roads.features, roads.min_zoom, roads.max_zoom) == ("ruas", 2, 8, 14)
    assert roads.sha256 == file_sha256(out / "jalan.pmtiles")
    assert roads.size == (out / "jalan.pmtiles").stat().st_size
    assert set(manifest.sources) == {tiles.ROADS_SOURCE, tiles.VILLAGES_SOURCE}
    assert fake.args[0][0] == str(tmp_path / "tippecanoe")


def test_build_keeps_edge_id_as_feature_id_and_attribute(monkeypatch, tmp_path, fake):
    _build(monkeypatch, tmp_path)
    road = fake.inputs["jalan.pmtiles"][1]
    assert road["id"] == 1
    assert road["properties"] == {"edge_id": 1, "highway_type": "track", "is_bridge": True}
    village = fake.inputs["desa.pmtiles"][0]
    assert "id" not in village
    assert village["properties"] == {"village_id": "1114000001", "name": "Alpha"}


def test_build_reprojects_to_lon_lat(monkeypatch, tmp_path, fake):
    _build(monkeypatch, tmp_path, _processed(tmp_path / "processed", utm_roads=True))
    start = fake.inputs["jalan.pmtiles"][0]["geometry"]["coordinates"][0]
    assert start == pytest.approx([98.0, 4.0], abs=1e-6)


def test_build_needs_the_bundle_of_the_same_version(monkeypatch, tmp_path, fake):
    def missing(root, version):
        raise DataNotLoadedError("no static data bundle")

    monkeypatch.setattr(tiles, "read_bundle", missing)
    with pytest.raises(InvalidInputError, match="build the bundle first"):
        tiles.build(VERSION, _processed(tmp_path / "p"), tmp_path / "static", tmp_path / "tiles")


def test_build_refuses_sources_that_changed_since_the_bundle(monkeypatch, tmp_path, fake):
    with pytest.raises(InvalidInputError, match=r"road_segments\.parquet .* changed since bundle"):
        _build(monkeypatch, tmp_path, **{tiles.ROADS_SOURCE: "0" * 64})


def test_build_refuses_missing_sources(monkeypatch, tmp_path, fake):
    processed = _processed(tmp_path / "processed")
    monkeypatch.setattr(tiles, "read_bundle", _bundle_of(processed))
    (processed / tiles.VILLAGES_SOURCE).unlink()
    with pytest.raises(InvalidInputError, match=r"villages\.parquet"):
        tiles.build(VERSION, processed, tmp_path / "static", tmp_path / "tiles")


def test_build_never_overwrites_a_version(monkeypatch, tmp_path, fake):
    _build(monkeypatch, tmp_path)
    with pytest.raises(InvalidInputError, match="already built"):
        _build(monkeypatch, tmp_path, _processed(tmp_path / "processed"))


def test_oversized_roads_are_refused_and_cleaned_up(monkeypatch, tmp_path, fake):
    monkeypatch.setattr(tiles, "MAX_ROADS_BYTES", 10)
    with pytest.raises(InvalidInputError, match="over 10"):
        _build(monkeypatch, tmp_path)
    assert list((tmp_path / "tiles").iterdir()) == []


def test_build_rejects_bad_versions(tmp_path):
    with pytest.raises(InvalidInputError, match="release version"):
        tiles.build("latest", tmp_path, tmp_path, tmp_path)
