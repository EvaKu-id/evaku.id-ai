import pandas as pd
import pytest

from evaku_ai.core.errors import InvalidInputError
from evaku_ai.static_data.builders.village_features import FEATURE_COLUMNS, build_village_features
from tests.fixtures.processed import UTM, write_village_inputs


def _write(tmp_path, **kwargs):
    return write_village_inputs(tmp_path, **kwargs)


def test_features_follow_the_geometry_and_join_the_other_tables(tmp_path):
    table = build_village_features(_write(tmp_path), UTM)
    assert table.columns.tolist() == FEATURE_COLUMNS
    alpha, beta = (table.iloc[0], table.iloc[1])
    assert (alpha["village_id"], alpha["name"], alpha["district_id"]) == (
        "1114000001",
        "Alpha",
        "1114000",
    )
    assert alpha["area_km2"] == pytest.approx(4.0, abs=0.01)
    assert alpha["basin_share"] == pytest.approx(0.5, abs=0.001)
    assert beta["basin_share"] == pytest.approx(0.0, abs=0.001)
    assert alpha["distance_to_stream_m"] == pytest.approx(1500.0, abs=0.5)
    assert beta["distance_to_stream_m"] == pytest.approx(3500.0, abs=0.5)
    assert alpha["population"] == 1000.0 and alpha["hand_median"] == 1.23
    assert (beta["population_hand_le_1m"], beta["population_hand_le_3m"]) == (0.0, 10.0)
    assert table["village_id"].map(type).eq(str).all()


def test_a_village_without_population_is_rejected(tmp_path):
    population = pd.DataFrame(
        {
            "village_id": ["1114000001"],
            "population": [1.0],
            "children_under5": [0.0],
            "elderly": [0.0],
        }
    )
    with pytest.raises(InvalidInputError, match="missing values"):
        build_village_features(_write(tmp_path, population=population), UTM)


def test_duplicate_village_ids_are_rejected(tmp_path):
    hand = pd.DataFrame(
        {
            "village_id": ["1114000001", "1114000001"],
            "hand_median": [1.0, 2.0],
            "population_hand_le_1m": [0.0, 0.0],
            "population_hand_le_3m": [0.0, 0.0],
        }
    )
    with pytest.raises(InvalidInputError, match="unique in the hand table"):
        build_village_features(_write(tmp_path, hand=hand), UTM)


def test_empty_streams_are_rejected(tmp_path):
    with pytest.raises(InvalidInputError, match="must not be empty"):
        build_village_features(_write(tmp_path, streams=False), UTM)
