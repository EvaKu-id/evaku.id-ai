from collections.abc import Sequence

import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry

from evaku_ai.core.errors import InvalidInputError
from evaku_ai.domain.arrays import BoolArray, FloatArray, IntArray

# Six decimals of a degree is about 0.1 m, enough for a route drawn on a map.
ROUTE_PRECISION_DEG = 1e-6


def crossing_points(
    lines: Sequence[BaseGeometry], streams: Sequence[BaseGeometry]
) -> tuple[FloatArray, FloatArray, IntArray]:
    line_array = np.asarray(lines, dtype=object)
    stream_array = np.asarray(streams, dtype=object)
    if not len(line_array) or not len(stream_array):
        empty = np.empty(0)
        return empty, empty, np.empty(0, dtype=np.int64)
    line_idx, stream_idx = shapely.STRtree(stream_array).query(line_array, predicate="intersects")
    shared = shapely.intersection(line_array[line_idx], stream_array[stream_idx])
    coords, pair = shapely.get_coordinates(shared, return_index=True)
    return coords[:, 0], coords[:, 1], line_idx[pair].astype(np.int64)


def near_crossings(
    x: FloatArray,
    y: FloatArray,
    sample_line: IntArray,
    crossing_x: FloatArray,
    crossing_y: FloatArray,
    crossing_line: IntArray,
    buffer_m: float,
) -> BoolArray:
    near = np.zeros(x.shape[0], dtype=bool)
    if not crossing_x.size or not x.size:
        return near
    tree = shapely.STRtree(shapely.points(crossing_x, crossing_y))
    sample, crossing = tree.query(shapely.points(x, y), predicate="dwithin", distance=buffer_m)
    # Only a crossing on the same road counts; a stream crossing a neighbouring road does not.
    same = sample_line[sample] == crossing_line[crossing]
    near[sample[same]] = True
    return near


def weighted_median(
    values: FloatArray, weights: FloatArray, groups: IntArray, n_groups: int
) -> FloatArray:
    out = np.full(n_groups, np.nan)
    usable = np.isfinite(values) & (weights > 0)
    v, w, g = values[usable], weights[usable], groups[usable]
    if not v.size:
        return out
    order = np.lexsort((v, g))
    v, w, g = v[order], w[order], g[order]
    total = np.bincount(g, weights=w, minlength=n_groups)
    before = np.concatenate([[0.0], np.cumsum(total)[:-1]])
    within = np.cumsum(w) - before[g]
    reached = np.flatnonzero(within >= 0.5 * total[g])
    first_group, first = np.unique(g[reached], return_index=True)
    out[first_group] = v[reached[first]]
    return out


def route_geojson(lines: Sequence[BaseGeometry]) -> dict[str, object] | None:
    # A site on the village's own origin node has no route to draw.
    if not len(lines):
        return None
    merged = shapely.line_merge(shapely.union_all(np.asarray(lines, dtype=object)))
    if merged.geom_type != "LineString":
        raise InvalidInputError(f"route edges do not form one line ({merged.geom_type})")
    coordinates = shapely.get_coordinates(shapely.set_precision(merged, ROUTE_PRECISION_DEG))
    return {"type": "LineString", "coordinates": coordinates.tolist()}
