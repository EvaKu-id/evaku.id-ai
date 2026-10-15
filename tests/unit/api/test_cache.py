import pytest

from evaku_ai.api.cache import SimulationCache, param_hash
from evaku_ai.api.schemas import SimulationRequest
from evaku_ai.domain.enums import Vehicle
from evaku_ai.engine.simulation import run_simulation
from tests.fixtures.small_graph import small_scenario


@pytest.fixture(scope="module")
def result():
    return run_simulation(small_scenario(), "r1", 1.0, Vehicle.CAR)


def request(**changes):
    return SimulationRequest(**({"river_id": "r1", "water_level_m": 1.0} | changes))


def test_param_hash_is_short_and_stable():
    key = param_hash("d1", request())
    assert len(key) == 16
    assert key == param_hash("d1", request())


def test_param_hash_changes_with_data_version_and_inputs():
    base = param_hash("d1", request())
    assert param_hash("d2", request()) != base
    assert param_hash("d1", request(vehicle="truck")) != base
    assert param_hash("d1", request(river_id="r2")) != base
    assert param_hash("d1", request(hub_ids=[2, 1])) == param_hash("d1", request(hub_ids=[1, 2]))
    assert param_hash("d1", request(hub_ids=[])) == base


def test_cache_evicts_least_recently_used(result):
    cache = SimulationCache(2)
    cache.put("a", result)
    cache.put("b", result)
    assert cache.get("a") is result
    cache.put("c", result)
    assert cache.get("b") is None
    assert cache.get("a") is result
    assert cache.get("c") is result


def test_cache_of_one_keeps_latest(result):
    cache = SimulationCache(1)
    cache.put("a", result)
    cache.put("a", result)
    cache.put("b", result)
    assert cache.get("a") is None
    assert cache.get("b") is result


def test_settings_reject_empty_cache():
    from evaku_ai.core.config import Settings

    with pytest.raises(ValueError, match="simulation_cache_size"):
        Settings(_env_file=None, simulation_cache_size=0)
