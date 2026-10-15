import numpy as np
import pytest

from evaku_ai.core.errors import InvalidInputError
from evaku_ai.domain.rules import ScoreWeights
from evaku_ai.engine.scoring import COMPONENTS, priority_scores

ELIGIBLE = np.array([True, True, True, False])


def all_components(**overrides):
    base = {
        "isolated_population": np.array([300.0, 100.0, 0.0, 900.0]),
        "vulnerable_share": np.array([0.4, 0.2, 0.0, 0.9]),
        "no_health_facility": np.array([1.0, 0.5, 0.0, 1.0]),
        "isolation_duration": np.array([48.0, 12.0, 0.0, 72.0]),
        "access_difficulty": np.array([1.0, 0.5, 0.0, 1.0]),
    }
    return base | overrides


def test_components_match_weight_fields():
    assert COMPONENTS == (
        "isolated_population",
        "vulnerable_share",
        "no_health_facility",
        "isolation_duration",
        "access_difficulty",
    )


def test_min_max_per_run_spans_zero_to_hundred():
    result = priority_scores(all_components(), ELIGIBLE)
    middle = 0.35 / 3 + 0.20 * 0.5 + 0.20 * 0.5 + 0.15 * 0.25 + 0.10 * 0.5
    np.testing.assert_allclose(result.score[:3], [100.0, 100.0 * middle, 0.0])


def test_village_without_isolation_scores_zero():
    result = priority_scores(all_components(), ELIGIBLE)
    assert result.score[3] == 0.0
    assert all(result.value[name][3] == 0.0 for name in COMPONENTS)
    assert result.raw["isolated_population"][3] == 900.0


def test_more_isolated_people_ranks_higher():
    components = {
        "isolated_population": np.array([200.0, 100.0, 50.0]),
        "vulnerable_share": np.array([0.3, 0.3, 0.3]),
    }
    result = priority_scores(components, np.ones(3, dtype=bool))
    assert result.score[0] > result.score[1] > result.score[2]


def test_single_eligible_village_gets_full_value_for_nonzero_need():
    components = all_components(vulnerable_share=np.array([0.0, 0.2, 0.0, 0.9]))
    result = priority_scores(components, np.array([True, False, False, False]))
    assert result.value["vulnerable_share"][0] == 0.0
    assert result.value["isolated_population"][0] == 1.0
    assert result.score[0] == pytest.approx(80.0)


def test_missing_component_weight_is_redistributed():
    components = all_components()
    del components["isolation_duration"]
    result = priority_scores(components, ELIGIBLE)
    assert result.weight["isolation_duration"] == 0.0
    assert sum(result.weight.values()) == pytest.approx(1.0)
    assert result.weight["isolated_population"] == pytest.approx(0.35 / 0.85)
    assert result.score[0] == pytest.approx(100.0)


def test_breakdown_explains_the_score():
    result = priority_scores(all_components(), ELIGIBLE)
    breakdown = result.breakdown(1)
    assert set(breakdown) == set(COMPONENTS)
    assert breakdown["isolated_population"] == pytest.approx(
        {"raw": 100.0, "value": 1 / 3, "weight": 0.35}
    )
    total = sum(part["weight"] * part["value"] for part in breakdown.values())
    assert result.score[1] == pytest.approx(100.0 * total)


def test_custom_weights_are_used():
    weights = ScoreWeights(
        isolated_population=1.0,
        vulnerable_share=0.0,
        no_health_facility=0.0,
        isolation_duration=0.0,
        access_difficulty=0.0,
    )
    result = priority_scores(all_components(), ELIGIBLE, weights)
    np.testing.assert_allclose(result.score[:3], [100.0, 100.0 / 3, 0.0])


def test_no_eligible_village_scores_all_zero():
    result = priority_scores(all_components(), np.zeros(4, dtype=bool))
    np.testing.assert_array_equal(result.score, np.zeros(4))


@pytest.mark.parametrize(
    ("components", "eligible", "message"),
    [
        ({}, ELIGIBLE, "at least one"),
        ({"speed": np.zeros(4)}, ELIGIBLE, "unknown"),
        ({"vulnerable_share": np.zeros(3)}, ELIGIBLE, "length"),
        ({"vulnerable_share": np.array([0.1, np.nan, 0.0, 0.0])}, ELIGIBLE, "finite"),
        ({"vulnerable_share": np.array([0.1, -1.0, 0.0, 0.0])}, ELIGIBLE, "finite"),
        ({"vulnerable_share": np.zeros(4)}, np.ones(4), "eligible"),
    ],
)
def test_rejects_invalid_components(components, eligible, message):
    with pytest.raises(InvalidInputError, match=message):
        priority_scores(components, eligible)


def test_rejects_components_without_weight():
    weights = ScoreWeights(
        isolated_population=1.0,
        vulnerable_share=0.0,
        no_health_facility=0.0,
        isolation_duration=0.0,
        access_difficulty=0.0,
    )
    with pytest.raises(InvalidInputError, match="no weight"):
        priority_scores({"vulnerable_share": np.zeros(4)}, ELIGIBLE, weights)
