import numpy as np
import pytest
from shapely.geometry import LineString

from evaku_ai.core.errors import InvalidInputError
from evaku_ai.geo.vector import crossing_points, near_crossings, route_geojson, weighted_median


def test_crossing_points_only_where_lines_cross():
    roads = [
        LineString([(0, 0), (100, 0)]),
        LineString([(0, 10), (100, 10)]),
        LineString([(0, 50), (100, 50)]),
    ]
    streams = [LineString([(50, -20), (50, 20)]), LineString([(0, 50), (0, 80)])]
    x, y, line = crossing_points(roads, streams)
    found = sorted(zip(line.tolist(), x.tolist(), y.tolist(), strict=True))
    assert found == [(0, 50.0, 0.0), (1, 50.0, 10.0), (2, 0.0, 50.0)]


def test_crossing_points_empty_inputs():
    x, _, line = crossing_points([], [LineString([(0, 0), (1, 1)])])
    assert x.size == 0 and line.dtype == np.int64
    assert crossing_points([LineString([(0, 0), (1, 1)])], [])[0].size == 0


def test_near_crossings_only_on_the_same_road():
    x = np.array([40.0, 80.0, 50.0])
    y = np.array([0.0, 0.0, 10.0])
    sample_line = np.array([0, 0, 1])
    near = near_crossings(x, y, sample_line, np.array([50.0]), np.array([0.0]), np.array([0]), 15.0)
    assert near.tolist() == [True, False, False]
    empty = near_crossings(x, y, sample_line, np.empty(0), np.empty(0), np.empty(0, int), 15.0)
    assert not empty.any()


def test_weighted_median():
    values = np.array([1.0, 2.0, 3.0, 10.0, 20.0, np.nan, 5.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 9.0, 5.0, 0.0])
    groups = np.array([0, 0, 0, 1, 1, 1, 2])
    result = weighted_median(values, weights, groups, 4)
    assert result[:2].tolist() == [2.0, 20.0]
    assert np.isnan(result[2:]).all()


def test_weighted_median_without_usable_values():
    result = weighted_median(np.array([np.nan]), np.array([1.0]), np.array([0]), 2)
    assert np.isnan(result).all()


@pytest.mark.parametrize(("weights", "expected"), [([1.0, 1.0], 1.0), ([1.0, 3.0], 4.0)])
def test_weighted_median_ties(weights, expected):
    result = weighted_median(np.array([1.0, 4.0]), np.array(weights), np.array([0, 0]), 1)
    assert result[0] == expected


def test_route_geojson_joins_edges_in_any_direction():
    edges = [LineString([(98.0, 4.0), (98.1, 4.0)]), LineString([(98.2, 4.1), (98.1, 4.0)])]
    route = route_geojson(edges)
    assert route["type"] == "LineString"
    coords = route["coordinates"]
    assert {tuple(coords[0]), tuple(coords[-1])} == {(98.0, 4.0), (98.2, 4.1)}
    assert len(coords) == 3


def test_route_geojson_rounds_coordinates():
    route = route_geojson([LineString([(98.12345678, 4.0), (98.2, 4.0)])])
    assert route["coordinates"][0][0] == pytest.approx(98.123457, abs=1e-9)


def test_route_geojson_without_edges_is_none():
    assert route_geojson([]) is None


def test_route_geojson_rejects_broken_routes():
    edges = [LineString([(0, 0), (1, 0)]), LineString([(5, 5), (6, 5)])]
    with pytest.raises(InvalidInputError, match="one line"):
        route_geojson(edges)
