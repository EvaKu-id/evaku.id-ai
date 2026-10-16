import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import pytest
from shapely.geometry import LineString, box

from evaku_ai.adapters import tippecanoe
from evaku_ai.adapters.download import file_sha256
from evaku_ai.core.errors import ExternalToolError
from evaku_ai.static_data.builders import tiles

pytestmark = pytest.mark.integration

VERSION = "2026-10-16.3"
ZOOM = 14
DECODE_TIMEOUT_S = 60.0


def _binary(name: str) -> Path:
    try:
        return tippecanoe.find_binary(name)
    except ExternalToolError as exc:
        pytest.fail(f"{exc}; run make tippecanoe")


def _tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    n = 2**zoom
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return int((lon + 180) / 360 * n), int(y)


def test_real_tiles_keep_edge_id_as_feature_id(tmp_path, monkeypatch):
    _binary("tippecanoe")
    decode = _binary("tippecanoe-decode")
    processed = tmp_path / "processed"
    processed.mkdir()
    gpd.GeoDataFrame(
        {"edge_id": [7, 8], "highway_type": ["primary", "track"], "is_bridge": [False, True]},
        geometry=[
            LineString([(98.05, 4.27), (98.06, 4.28)]),
            LineString([(98.06, 4.28), (98.065, 4.285)]),
        ],
        crs=4326,
    ).to_parquet(processed / tiles.ROADS_SOURCE)
    gpd.GeoDataFrame(
        {"village_id": ["1114000001"], "name": ["Alpha"]},
        geometry=[box(98.05, 4.27, 98.07, 4.29)],
        crs=4326,
    ).to_parquet(processed / tiles.VILLAGES_SOURCE)
    sources = {name: file_sha256(processed / name) for name in tiles.SOURCES.values()}
    bundle = SimpleNamespace(manifest=SimpleNamespace(sources=sources))
    monkeypatch.setattr(tiles, "read_bundle", lambda root, version: bundle)

    tiles.build(VERSION, processed, tmp_path / "static", tmp_path / "tiles")

    x, y = _tile(98.06, 4.28, ZOOM)
    done = subprocess.run(  # noqa: S603
        [
            str(decode),
            str(tmp_path / "tiles" / VERSION / "jalan.pmtiles"),
            str(ZOOM),
            str(x),
            str(y),
        ],
        capture_output=True,
        text=True,
        timeout=DECODE_TIMEOUT_S,
        check=True,
    )
    layers = json.loads(done.stdout)["features"]
    assert [layer["properties"]["layer"] for layer in layers] == ["ruas"]
    features = layers[0]["features"]
    assert {f["id"] for f in features} == {7, 8}
    for feature in features:
        assert feature["id"] == feature["properties"]["edge_id"]
        assert set(feature["properties"]) == {"edge_id", "highway_type", "is_bridge"}
