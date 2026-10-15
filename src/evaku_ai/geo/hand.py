import itertools
import math
from collections.abc import Sequence

import numpy as np
import shapely
from scipy import ndimage
from shapely.geometry.base import BaseGeometry

from evaku_ai.core.errors import InvalidInputError
from evaku_ai.domain.arrays import BoolArray, FloatArray, IntArray

M2_PER_KM2 = 1_000_000.0

# WhiteboxTools D8 pointer codes (non-ESRI) to (row, col) steps; 0 marks a cell with no outflow.
D8_STEPS: dict[int, tuple[int, int]] = {
    1: (-1, 1),
    2: (0, 1),
    4: (1, 1),
    8: (1, 0),
    16: (1, -1),
    32: (0, -1),
    64: (-1, -1),
    128: (-1, 0),
}


def sea_mask(dem: FloatArray, nodata: float | None, keep: BoolArray | None = None) -> BoolArray:
    data = np.asarray(dem, dtype=np.float64)
    low = ~np.isfinite(data) | (data <= 0)
    if nodata is not None:
        low |= data == nodata
    if keep is not None:
        # Estuaries sit at sea level; mapped river cells stay land so channels reach the coast.
        low &= ~keep
    labels, _ = ndimage.label(low)
    border = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    edge_labels = np.unique(border[border > 0])
    return np.isin(labels, edge_labels)


def threshold_cells(area_km2: float, cell_m: float) -> int:
    if not (area_km2 > 0 and cell_m > 0):
        raise InvalidInputError(f"need area_km2 > 0 and cell_m > 0, got {area_km2}, {cell_m}")
    return max(1, math.ceil(area_km2 * M2_PER_KM2 / cell_m**2))


def snap_pour_point(
    accumulation: FloatArray, row: int, col: int, radius_cells: int
) -> tuple[int, int]:
    n_rows, n_cols = accumulation.shape
    if not (0 <= row < n_rows and 0 <= col < n_cols):
        raise InvalidInputError(f"pour point ({row}, {col}) is outside the raster")
    top, left = max(row - radius_cells, 0), max(col - radius_cells, 0)
    window = np.asarray(accumulation[top : row + radius_cells + 1, left : col + radius_cells + 1])
    window = np.where(np.isfinite(window), window, -np.inf)
    best_row, best_col = np.unravel_index(int(np.argmax(window)), window.shape)
    return top + int(best_row), left + int(best_col)


def trace_downstream(pointer: IntArray, land: BoolArray, row: int, col: int) -> tuple[int, int]:
    n_rows, n_cols = pointer.shape
    for _ in range(pointer.size):
        code = int(pointer[row, col])
        if code == 0:
            return row, col
        if code not in D8_STEPS:
            raise InvalidInputError(f"invalid D8 code {code} at ({row}, {col})")
        d_row, d_col = D8_STEPS[code]
        next_row, next_col = row + d_row, col + d_col
        inside = 0 <= next_row < n_rows and 0 <= next_col < n_cols
        if not inside or not land[next_row, next_col]:
            return row, col
        row, col = next_row, next_col
    raise InvalidInputError("flow path loops; the D8 pointer is inconsistent")


def touches_edge(mask: BoolArray) -> bool:
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def median_by_distance(
    hand: FloatArray,
    streams: BoolArray,
    valid: BoolArray,
    bands_m: Sequence[float],
    cell_m: float,
) -> FloatArray:
    distance = ndimage.distance_transform_edt(~streams) * cell_m
    # The first band starts below zero so stream cells (distance 0) fall into it.
    edges = [-1.0, *bands_m]
    usable = valid & np.isfinite(hand)
    medians = np.full(len(bands_m), np.nan)
    for i, (low, high) in enumerate(itertools.pairwise(edges)):
        in_band = usable & (distance > low) & (distance <= high)
        if in_band.any():
            medians[i] = float(np.median(hand[in_band]))
    return medians


def stream_agreement(
    extracted: Sequence[BaseGeometry], reference: Sequence[BaseGeometry], match_m: float
) -> tuple[float, float]:
    return (
        _share_within(reference, extracted, match_m),
        _share_within(extracted, reference, match_m),
    )


def _share_within(
    lines: Sequence[BaseGeometry], target: Sequence[BaseGeometry], distance_m: float
) -> float:
    total = float(shapely.length(np.asarray(lines, dtype=object)).sum()) if len(lines) else 0.0
    if total == 0 or not len(target):
        return 0.0
    zone = shapely.union_all(shapely.buffer(np.asarray(target, dtype=object), distance_m))
    inside = shapely.intersection(np.asarray(lines, dtype=object), zone)
    return float(shapely.length(inside).sum()) / total


def fill_unreached(hand: FloatArray, dem: FloatArray, land: BoolArray) -> tuple[FloatArray, int]:
    # Coastal cells that drain straight into the sea never meet a stream; the sea is their drainage.
    out = np.where(land, np.asarray(hand, dtype=np.float64), np.nan)
    missing = land & ~np.isfinite(out)
    out[missing] = np.maximum(dem[missing], 0.0)
    return out, int(missing.sum())


def burn(dem: FloatArray, mask: BoolArray, depth_m: float) -> FloatArray:
    return np.where(mask & np.isfinite(dem), dem - depth_m, dem)


def drainage_target(pointer: IntArray, streams: BoolArray, land: BoolArray) -> IntArray:
    n_rows, n_cols = pointer.shape
    codes = np.asarray(pointer, dtype=np.int64).ravel()
    d_row = np.zeros(max(D8_STEPS) + 1, dtype=np.int64)
    d_col = np.zeros_like(d_row)
    for code, (dr, dc) in D8_STEPS.items():
        d_row[code], d_col[code] = dr, dc
    known = np.isin(codes, list(D8_STEPS))
    safe = np.where(known, codes, 0)
    index = np.arange(codes.size)
    rows, cols = index // n_cols + d_row[safe], index % n_cols + d_col[safe]
    inside = known & (rows >= 0) & (rows < n_rows) & (cols >= 0) & (cols < n_cols)
    downstream = np.where(
        inside, rows.clip(0, n_rows - 1) * n_cols + cols.clip(0, n_cols - 1), index
    )
    flat_streams, flat_land = streams.ravel(), land.ravel()
    downstream = np.where(flat_land[downstream] & ~flat_streams, downstream, index)
    # Pointer jumping: each pass doubles the path length followed, so log2(cells) passes suffice.
    for _ in range(max(1, math.ceil(math.log2(codes.size))) + 1):
        jumped = downstream[downstream]
        if np.array_equal(jumped, downstream):
            break
        downstream = jumped
    target = np.where(flat_streams[downstream] & flat_land, downstream, -1)
    return target.reshape(pointer.shape)


def restore_burn(
    hand: FloatArray, burned_m: FloatArray, target: IntArray
) -> tuple[FloatArray, int]:
    # Burning lowers both the cell and its stream; add back the cell's depth, remove the stream's.
    flat_burn = burned_m.ravel()
    reached = target >= 0
    lift = np.where(reached, flat_burn[np.where(reached, target, 0)].reshape(target.shape), 0.0)
    out = np.asarray(hand, dtype=np.float64) + burned_m - lift
    negative = np.isfinite(out) & (out < 0)
    out[negative] = 0.0
    return out, int(negative.sum())
