import numpy as np
import pytest
from shapely.geometry import LineString

from evaku_ai.core.errors import InvalidInputError
from evaku_ai.geo.hand import (
    burn,
    drainage_target,
    fill_unreached,
    median_by_distance,
    restore_burn,
    sea_mask,
    snap_pour_point,
    stream_agreement,
    threshold_cells,
    touches_edge,
    trace_downstream,
)


def test_sea_mask_keeps_inland_depressions():
    dem = np.array(
        [
            [0.0, 0.0, 5.0, 5.0, 5.0],
            [-1.0, 3.0, 5.0, -2.0, 5.0],
            [0.0, 3.0, 5.0, 5.0, 5.0],
            [-9.0, 4.0, 6.0, 6.0, 6.0],
        ]
    )
    mask = sea_mask(dem, nodata=-9.0)
    assert mask[:, 0].all()
    assert mask[0, 1]
    assert not mask[1, 3]
    assert mask.sum() == 5


def test_sea_mask_keeps_mapped_river_cells_as_land():
    dem = np.array([[5.0, 0.0, 5.0], [5.0, 0.0, 5.0], [0.0, 0.0, 0.0]])
    keep = np.array([[False, True, False], [False, True, False], [False, False, False]])
    mask = sea_mask(dem, nodata=None, keep=keep)
    assert mask[2].all()
    assert not mask[:2].any()


def test_sea_mask_without_nodata_treats_nan_as_sea():
    dem = np.array([[np.nan, 2.0], [2.0, 2.0]])
    assert sea_mask(dem, nodata=None).tolist() == [[True, False], [False, False]]


@pytest.mark.parametrize(("km2", "expected"), [(1.0, 1112), (0.0001, 1), (0.0009, 1)])
def test_threshold_cells(km2, expected):
    assert threshold_cells(km2, 30.0) == expected


@pytest.mark.parametrize(("km2", "cell"), [(0.0, 30.0), (1.0, 0.0), (-1.0, 30.0)])
def test_threshold_cells_rejects_non_positive(km2, cell):
    with pytest.raises(InvalidInputError):
        threshold_cells(km2, cell)


def test_snap_pour_point_moves_to_highest_accumulation():
    acc = np.zeros((6, 6))
    acc[4, 5] = 50.0
    acc[1, 1] = 99.0
    assert snap_pour_point(acc, 5, 5, radius_cells=1) == (4, 5)
    assert snap_pour_point(acc, 0, 0, radius_cells=2) == (1, 1)


def test_snap_pour_point_ignores_nan_and_rejects_outside():
    acc = np.array([[np.nan, 1.0], [2.0, np.nan]])
    assert snap_pour_point(acc, 0, 0, radius_cells=3) == (1, 0)
    with pytest.raises(InvalidInputError, match="outside"):
        snap_pour_point(acc, 2, 0, radius_cells=1)


def test_trace_downstream_stops_before_sea():
    pointer = np.array([[8, 0, 0], [8, 0, 0], [2, 2, 2]])
    land = np.array([[True, True, True], [True, True, True], [True, True, False]])
    assert trace_downstream(pointer, land, 0, 0) == (2, 1)


def test_trace_downstream_stops_at_pit_and_edge():
    pointer = np.array([[2, 0], [128, 0]])
    land = np.ones((2, 2), dtype=bool)
    assert trace_downstream(pointer, land, 0, 0) == (0, 1)
    assert trace_downstream(pointer, land, 1, 0) == (0, 1)
    assert trace_downstream(np.array([[128]]), np.ones((1, 1), bool), 0, 0) == (0, 0)


@pytest.mark.parametrize(
    ("pointer", "message"),
    [(np.array([[2, 32]]), "loops"), (np.array([[3, 0]]), "invalid D8 code")],
)
def test_trace_downstream_rejects_bad_pointers(pointer, message):
    with pytest.raises(InvalidInputError, match=message):
        trace_downstream(pointer, np.ones(pointer.shape, bool), 0, 0)


def test_touches_edge():
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    assert not touches_edge(mask)
    mask[3, 2] = True
    assert touches_edge(mask)


def test_median_by_distance_rises_in_a_valley():
    cols = np.arange(41)
    hand = np.tile(np.abs(cols - 20) * 2.0, (5, 1))
    streams = np.zeros_like(hand, dtype=bool)
    streams[:, 20] = True
    medians = median_by_distance(hand, streams, np.ones_like(streams), (100.0, 300.0, 600.0), 30.0)
    assert medians[0] < medians[1] < medians[2]
    assert medians[0] == pytest.approx(4.0)


def test_median_by_distance_empty_band_is_nan():
    hand = np.zeros((3, 3))
    streams = np.ones((3, 3), dtype=bool)
    medians = median_by_distance(hand, streams, np.ones((3, 3), bool), (100.0, 500.0), 30.0)
    assert medians[0] == 0.0
    assert np.isnan(medians[1])


def test_stream_agreement():
    river = LineString([(0, 0), (1000, 0)])
    assert stream_agreement([river], [river], 90.0) == pytest.approx((1.0, 1.0))
    shifted = LineString([(0, 200), (1000, 200)])
    assert stream_agreement([shifted], [river], 90.0) == (0.0, 0.0)
    half = LineString([(0, 0), (500, 0)])
    recall, precision = stream_agreement([half], [river], 90.0)
    assert recall == pytest.approx(0.59, abs=0.01)
    assert precision == pytest.approx(1.0)
    assert stream_agreement([], [river], 90.0) == (0.0, 0.0)


def test_fill_unreached_uses_height_above_sea():
    hand = np.array([[np.nan, 1.0], [2.0, np.nan]])
    dem = np.array([[3.0, 9.0], [9.0, -0.5]])
    land = np.array([[True, True], [True, False]])
    filled, count = fill_unreached(hand, dem, land)
    assert count == 1
    assert filled[0].tolist() == [3.0, 1.0]
    assert filled[1, 0] == 2.0
    assert np.isnan(filled[1, 1])


def test_burn_lowers_masked_land_only():
    dem = np.array([[5.0, np.nan], [3.0, 4.0]])
    mask = np.array([[True, True], [False, True]])
    out = burn(dem, mask, 2.0)
    assert out[0, 0] == 3.0 and np.isnan(out[0, 1])
    assert out[1].tolist() == [3.0, 2.0]


def test_drainage_target_follows_flow_to_the_first_stream():
    # Row 0 flows east into column 3; column 3 flows south; stream starts at (2, 3).
    pointer = np.array([[2, 2, 2, 8], [2, 2, 2, 8], [0, 0, 0, 8], [0, 0, 0, 0]])
    streams = np.zeros((4, 4), dtype=bool)
    streams[2:, 3] = True
    land = np.ones((4, 4), dtype=bool)
    target = drainage_target(pointer, streams, land)
    assert target[0, 0] == target[1, 2] == 2 * 4 + 3
    assert target[3, 3] == 3 * 4 + 3
    assert target[2, 0] == -1


def test_drainage_target_stops_at_sea_edges_and_loops():
    pointer = np.array([[2, 32, 8], [0, 0, 2]])
    streams = np.zeros((2, 3), dtype=bool)
    land = np.array([[True, True, True], [True, True, True]])
    assert drainage_target(pointer, streams, land).tolist() == [[-1, -1, -1], [-1, -1, -1]]
    land[1, 2] = False
    streams[0, 2] = True
    target = drainage_target(np.array([[2, 2, 8], [0, 0, 8]]), streams, land)
    assert target[0].tolist() == [2, 2, 2]
    assert target[1, 2] == -1


def test_restore_burn_removes_the_burn_from_hand():
    hand = np.array([[0.0, 12.0, 4.0, np.nan]])
    burned = np.array([[10.0, 0.0, 10.0, 0.0]])
    target = np.array([[0, 0, 0, -1]])
    out, clipped = restore_burn(hand, burned, target)
    assert out[0, :3].tolist() == [0.0, 2.0, 4.0]
    assert np.isnan(out[0, 3])
    assert clipped == 0
    out, clipped = restore_burn(np.array([[0.0, 6.0]]), np.array([[10.0, 0.0]]), np.array([[0, 0]]))
    assert out.tolist() == [[0.0, 0.0]]
    assert clipped == 1


def test_drainage_target_gives_up_on_odd_cycles():
    pointer = np.array([[2, 8], [0, 64]])
    target = drainage_target(pointer, np.zeros((2, 2), dtype=bool), np.ones((2, 2), dtype=bool))
    assert (target == -1).all()
