import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from evaku_ai.adapters.raster import read_depth_raster
from evaku_ai.core.errors import InvalidInputError


def _write(path, crs="EPSG:32647"):
    values = np.array([[0.0, 0.5], [-9999.0, 1.2]], dtype=np.float32)
    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": from_origin(400000.0, 460000.0, 30.0, 30.0),
        "nodata": -9999.0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values, 1)
    return path


def test_read_depth_raster_turns_nodata_into_nan(tmp_path):
    raster = read_depth_raster(_write(tmp_path / "depth.tif"), 32647)
    assert raster.values[0].tolist() == [0.0, 0.5]
    assert np.isnan(raster.values[1, 0])
    assert raster.transform == (30.0, 0.0, 400000.0, 0.0, -30.0, 460000.0)


def test_read_depth_raster_rejects_other_crs(tmp_path):
    with pytest.raises(InvalidInputError, match="EPSG:32647"):
        read_depth_raster(_write(tmp_path / "depth.tif", crs="EPSG:4326"), 32647)


def test_read_depth_raster_rejects_large_files(tmp_path):
    with pytest.raises(InvalidInputError, match="larger than"):
        read_depth_raster(_write(tmp_path / "depth.tif"), 32647, max_bytes=10)
