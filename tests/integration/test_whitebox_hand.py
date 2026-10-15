import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from evaku_ai.adapters import whitebox
from evaku_ai.adapters.download import fetch

pytestmark = pytest.mark.integration

SIZE = 60
VALLEY_COL = 30


def _dem(path):
    rows, cols = np.mgrid[0:SIZE, 0:SIZE]
    values = 10.0 + 0.5 * np.abs(cols - VALLEY_COL) + 0.1 * (SIZE - rows)
    profile = {
        "driver": "GTiff",
        "height": SIZE,
        "width": SIZE,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32647",
        "transform": from_origin(400000.0, 450000.0, 30.0, 30.0),
        "nodata": -9999.0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values.astype(np.float32), 1)
    return path


def test_real_whitebox_hand_is_zero_on_the_valley_floor(tmp_path):
    tool = whitebox.WHITEBOX_TOOLS
    binary = whitebox.install(fetch(tool), tmp_path / "bin")
    assert whitebox.version(binary).startswith("WhiteboxTools v2.4")
    dem = _dem(tmp_path / "dem.tif")
    breached = whitebox.breach(binary, dem, tmp_path / "breached.tif", 10)
    pointer = whitebox.d8_pointer(binary, breached, tmp_path / "d8.tif")
    acc = whitebox.flow_accumulation(binary, pointer, tmp_path / "acc.tif")
    streams = whitebox.extract_streams(binary, acc, tmp_path / "streams.tif", 100)
    hand_path = whitebox.elevation_above_stream(binary, breached, streams, tmp_path / "hand.tif")
    with rasterio.open(hand_path) as src:
        hand = src.read(1, masked=True)
    row = hand[SIZE // 2]
    assert row[VALLEY_COL] == pytest.approx(0.0, abs=1e-3)
    assert row[VALLEY_COL + 5] > row[VALLEY_COL + 1] > 0
    assert row[VALLEY_COL - 5] > row[VALLEY_COL - 1] > 0
