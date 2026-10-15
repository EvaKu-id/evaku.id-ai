from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

from evaku_ai.core.errors import InvalidInputError
from evaku_ai.domain.arrays import FloatArray
from evaku_ai.geo.raster import GridTransform

MAX_RASTER_BYTES = 2_000_000_000


@dataclass(frozen=True)
class DepthRaster:
    values: FloatArray
    transform: GridTransform


def read_depth_raster(path: Path, epsg: int, max_bytes: int = MAX_RASTER_BYTES) -> DepthRaster:
    if path.stat().st_size > max_bytes:
        raise InvalidInputError(f"{path.name} is larger than {max_bytes} bytes")
    with rasterio.open(path) as src:
        if src.crs is None or src.crs.to_epsg() != epsg:
            raise InvalidInputError(f"{path.name} must be in EPSG:{epsg}, got {src.crs}")
        values = src.read(1, masked=True).astype(np.float64).filled(np.nan)
        t = src.transform
    return DepthRaster(values, (t.a, t.b, t.c, t.d, t.e, t.f))
