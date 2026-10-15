import numpy as np

from evaku_ai.domain.enums import Vehicle, VillageAccessStatus
from evaku_ai.engine.simulation import run_simulation
from evaku_ai.engine.synthetic import RIVER_ID, synthetic_scenario


def test_synthetic_scenario_is_deterministic():
    first, second = synthetic_scenario(), synthetic_scenario()
    np.testing.assert_array_equal(first.graph.hand_min, second.graph.hand_min)
    np.testing.assert_array_equal(first.population.total, second.population.total)
    assert first.graph.is_bridge.sum() == 3
    assert len(first.village_ids) == 9


def test_dry_synthetic_scenario_has_no_isolation():
    result = run_simulation(synthetic_scenario(), RIVER_ID, 0.0, Vehicle.CAR)
    assert result.summary.isolated_villages == 0
    assert result.roads == []


def test_high_water_isolates_the_far_bank():
    result = run_simulation(synthetic_scenario(), RIVER_ID, 2.0, Vehicle.CAR)
    statuses = [v.status for v in result.villages]
    assert statuses.count(VillageAccessStatus.FULLY_ISOLATED) > 0
    assert statuses[0] is VillageAccessStatus.REACHABLE
    assert result.summary.isolated_population > 0
