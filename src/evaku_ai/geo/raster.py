from collections.abc import Sequence
from typing import Literal

import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry

from evaku_ai.domain.arrays import BoolArray, FloatArray, IntArray

# (a, b, c, d, e, f) of a north-up affine transform: x = a * col + c, y = e * row + f.
GridTransform = tuple[float, float, float, float, float, float]


def pixel_centers(
    values: FloatArray, transform: GridTransform, nodata: float | None
) -> tuple[FloatArray, FloatArray, FloatArray]:
    data = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(data) & (data > 0)
    if nodata is not None:
        valid &= data != nodata
    rows, cols = np.nonzero(valid)
    a, _, c, _, e, f = transform
    x = (c + (cols + 0.5) * a).astype(np.float64)
    y = (f + (rows + 0.5) * e).astype(np.float64)
    return x, y, data[rows, cols]


def sample_at(
    values: FloatArray,
    transform: GridTransform,
    x: FloatArray,
    y: FloatArray,
    nodata: float | None,
) -> FloatArray:
    data = np.asarray(values, dtype=np.float64)
    a, _, c, _, e, f = transform
    cols = np.floor((np.asarray(x) - c) / a).astype(np.int64)
    rows = np.floor((np.asarray(y) - f) / e).astype(np.int64)
    inside = (rows >= 0) & (rows < data.shape[0]) & (cols >= 0) & (cols < data.shape[1])
    out = np.zeros(cols.shape[0], dtype=np.float64)
    picked = data[rows[inside], cols[inside]]
    ok = np.isfinite(picked) & (picked >= 0)
    if nodata is not None:
        ok &= picked != nodata
    out[np.flatnonzero(inside)[ok]] = picked[ok]
    return out


def sample_values(
    values: FloatArray,
    transform: GridTransform,
    x: FloatArray,
    y: FloatArray,
    nodata: float | None,
) -> FloatArray:
    data = np.asarray(values, dtype=np.float64)
    a, _, c, _, e, f = transform
    cols = np.floor((np.asarray(x) - c) / a).astype(np.int64)
    rows = np.floor((np.asarray(y) - f) / e).astype(np.int64)
    inside = (rows >= 0) & (rows < data.shape[0]) & (cols >= 0) & (cols < data.shape[1])
    out = np.full(cols.shape[0], np.nan)
    picked = data[rows[inside], cols[inside]]
    if nodata is not None:
        picked = np.where(picked == nodata, np.nan, picked)
    out[inside] = picked
    return out


def line_samples(
    lines: Sequence[BaseGeometry], step_m: float
) -> tuple[FloatArray, FloatArray, IntArray]:
    dense = shapely.segmentize(np.asarray(lines, dtype=object), step_m)
    coords, index = shapely.get_coordinates(dense, return_index=True)
    return coords[:, 0].astype(np.float64), coords[:, 1].astype(np.float64), index.astype(np.int64)


def reduce_by_line(
    values: FloatArray, line_index: IntArray, n_lines: int, how: Literal["min", "max"]
) -> FloatArray:
    fill, reduce = (np.inf, np.fmin) if how == "min" else (-np.inf, np.fmax)
    out = np.full(n_lines, fill)
    reduce.at(out, line_index, values)
    return np.where(np.isfinite(out), out, np.nan)


def min_along_lines(
    values: FloatArray,
    transform: GridTransform,
    nodata: float | None,
    lines: Sequence[BaseGeometry],
    step_m: float,
    skip: BoolArray | None = None,
) -> FloatArray:
    return _along_lines(values, transform, nodata, lines, step_m, skip, "min")


def max_along_lines(
    values: FloatArray,
    transform: GridTransform,
    nodata: float | None,
    lines: Sequence[BaseGeometry],
    step_m: float,
    skip: BoolArray | None = None,
) -> FloatArray:
    return _along_lines(values, transform, nodata, lines, step_m, skip, "max")


def _along_lines(
    values: FloatArray,
    transform: GridTransform,
    nodata: float | None,
    lines: Sequence[BaseGeometry],
    step_m: float,
    skip: BoolArray | None,
    how: Literal["min", "max"],
) -> FloatArray:
    x, y, index = line_samples(lines, step_m)
    sampled = sample_values(values, transform, x, y, nodata)
    if skip is not None:
        sampled = np.where(skip, np.nan, sampled)
    return reduce_by_line(sampled, index, len(lines), how)
