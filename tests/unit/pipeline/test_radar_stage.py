import itertools

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString

from evaku_ai.domain.enums import RoadCondition
from evaku_ai.engine.graph import RoadGraph
from evaku_ai.pipeline.stages.radar import road_status_from_depth

SAFE, LIMITED, CUT, UNKNOWN = (int(c) for c in RoadCondition)
X0, Y0, CELL, SIZE = 400000.0, 460000.0, 30.0, 20
ROW_Y = Y0 - 5.5 * CELL
NODE_X = [X0 + 15, X0 + 165, X0 + 225, X0 + 375, X0 + 900]


def _graph():
    u = np.array([0, 1, 2, 3], dtype=np.int64)
    v = np.array([1, 2, 3, 4], dtype=np.int64)
    return RoadGraph(
        edge_u=u,
        edge_v=v,
        travel_s=np.ones(4),
        length_m=np.ones(4),
        hand_min=np.zeros(4),
        is_bridge=np.array([False, True, False, False]),
        node_x=np.array(NODE_X),
        node_y=np.full(5, ROW_Y),
    )


def _lines():
    return [LineString([(a, ROW_Y), (b, ROW_Y)]) for a, b in itertools.pairwise(NODE_X)]


def _raster(path, values):
    profile = {
        "driver": "GTiff",
        "height": SIZE,
        "width": SIZE,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32647",
        "transform": from_origin(X0, Y0, CELL, CELL),
        "nodata": -9999.0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.where(np.isfinite(values), values, -9999.0).astype(np.float32), 1)
    return path


def test_dry_raster_makes_every_covered_road_safe(tmp_path):
    path = _raster(tmp_path / "depth.tif", np.zeros((SIZE, SIZE)))
    status, depth = road_status_from_depth(path, _lines(), _graph(), 32647)
    assert status.tolist() == [SAFE, SAFE, SAFE, SAFE]
    assert depth.tolist() == [0.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize(("water_m", "expected"), [(0.25, SAFE), (0.5, LIMITED), (0.8, CUT)])
def test_flooded_road_and_bridge_follow_the_approach(tmp_path, water_m, expected):
    values = np.zeros((SIZE, SIZE))
    values[5, 2:4] = water_m
    status, depth = road_status_from_depth(
        _raster(tmp_path / "d.tif", values),
        _lines(),
        _graph(),
        32647,
    )
    assert status[0] == expected
    assert status[1] == expected
    assert depth[1] == pytest.approx(water_m * 100)
    assert status[2] == SAFE


def test_uncovered_road_is_unknown_and_partly_covered_uses_covered_part(tmp_path):
    values = np.zeros((SIZE, SIZE))
    values[:, 10:] = np.nan
    status, _ = road_status_from_depth(
        _raster(tmp_path / "d.tif", values), _lines(), _graph(), 32647
    )
    assert status[2] == SAFE
    assert status[3] == UNKNOWN
