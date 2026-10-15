import numpy as np
import pytest
from shapely.geometry import LineString

from evaku_ai.geo.raster import (
    line_samples,
    max_along_lines,
    min_along_lines,
    pixel_centers,
    reduce_by_line,
    sample_at,
    sample_values,
)

TRANSFORM = (0.5, 0.0, 10.0, 0.0, -0.5, 20.0)


def test_pixel_centers_skip_zero_nan_and_nodata():
    values = np.array([[1.0, 0.0], [-9.0, np.nan], [2.5, 3.0]])
    x, y, v = pixel_centers(values, TRANSFORM, nodata=-9.0)
    np.testing.assert_allclose(x, [10.25, 10.25, 10.75])
    np.testing.assert_allclose(y, [19.75, 18.75, 18.75])
    np.testing.assert_allclose(v, [1.0, 2.5, 3.0])


def test_pixel_centers_without_nodata():
    x, _, v = pixel_centers(np.array([[4.0]]), TRANSFORM, nodata=None)
    assert (x[0], v[0]) == (10.25, 4.0)


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [(10.1, 19.9, 1.0), (10.9, 18.6, 3.0), (9.0, 19.9, 0.0), (10.1, 21.0, 0.0), (10.1, 19.4, 0.0)],
)
def test_sample_at(x, y, expected):
    values = np.array([[1.0, 2.0], [-9.0, 5.0], [6.0, 3.0]])
    result = sample_at(values, TRANSFORM, np.array([x]), np.array([y]), nodata=-9.0)
    assert result[0] == expected


def test_sample_values_keeps_nan_for_missing_data():
    values = np.array([[1.0, 2.0], [-9.0, 0.0], [6.0, 3.0]])
    x = np.array([10.1, 10.1, 9.0, 10.9])
    y = np.array([19.9, 19.4, 19.9, 19.4])
    result = sample_values(values, TRANSFORM, x, y, nodata=-9.0)
    assert result[0] == 1.0
    assert np.isnan(result[1]) and np.isnan(result[2])
    assert result[3] == 0.0
    assert sample_values(values, TRANSFORM, x[:1], y[:1], nodata=None)[0] == 1.0


def test_line_samples_respects_step_and_keeps_ends():
    lines = [LineString([(0, 0), (100, 0)]), LineString([(0, 5), (0, 20)])]
    x, y, index = line_samples(lines, 15.0)
    first = index == 0
    assert x[first][0] == 0 and x[first][-1] == 100
    assert np.diff(x[first]).max() <= 15.0
    assert y[index == 1].tolist() == [5.0, 20.0]


def test_reduce_by_line_without_values_is_nan():
    values = np.array([3.0, 1.0, np.nan, 7.0])
    index = np.array([0, 0, 1, 2])
    assert reduce_by_line(values, index, 4, "min")[[0, 2]].tolist() == [1.0, 7.0]
    assert np.isnan(reduce_by_line(values, index, 4, "min")[[1, 3]]).all()
    assert reduce_by_line(values, index, 4, "max")[0] == 3.0


GRID = (10.0, 0.0, 0.0, 0.0, -10.0, 50.0)


def _valley():
    values = np.tile(np.abs(np.arange(5) - 2) * 2.0, (5, 1))
    values[0, 0] = -9.0
    return values


def test_min_and_max_along_lines():
    across = LineString([(1, 25), (49, 25)])
    along = LineString([(25, 49), (25, 1)])
    outside = LineString([(100, 100), (120, 100)])
    lines = [across, along, outside]
    lows = min_along_lines(_valley(), GRID, -9.0, lines, 5.0)
    highs = max_along_lines(_valley(), GRID, -9.0, lines, 5.0)
    assert lows[:2].tolist() == [0.0, 0.0]
    assert highs[:2].tolist() == [4.0, 0.0]
    assert np.isnan(lows[2]) and np.isnan(highs[2])


def test_min_along_lines_skips_masked_samples():
    across = LineString([(1, 25), (49, 25)])
    x, _, _ = line_samples([across], 5.0)
    skip = np.abs(x - 25) < 10
    assert min_along_lines(_valley(), GRID, -9.0, [across], 5.0, skip)[0] == 2.0
