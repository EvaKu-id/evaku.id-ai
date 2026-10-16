import hashlib
import json
from collections import OrderedDict
from threading import Lock

from evaku_ai.api.schemas import SimulationRequest
from evaku_ai.engine.simulation import ScenarioData, SimulationResult, run_simulation

HASH_CHARS = 16


def param_hash(data_version: str, request: SimulationRequest) -> str:
    params = {
        "data_version": data_version,
        "river_id": request.river_id,
        "water_level_m": request.water_level_m,
        "vehicle": request.vehicle.value,
        "hub_ids": sorted(set(request.hub_ids)) if request.hub_ids else None,
    }
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:HASH_CHARS]


class SimulationCache:
    def __init__(self, size: int) -> None:
        self._size = size
        self._items: OrderedDict[str, SimulationResult] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> SimulationResult | None:
        with self._lock:
            result = self._items.get(key)
            if result is not None:
                self._items.move_to_end(key)
            return result

    def put(self, key: str, result: SimulationResult) -> None:
        with self._lock:
            self._items[key] = result
            self._items.move_to_end(key)
            while len(self._items) > self._size:
                self._items.popitem(last=False)


def cached_simulation(
    cache: SimulationCache, scenario: ScenarioData, data_version: str, request: SimulationRequest
) -> tuple[str, SimulationResult]:
    key = param_hash(data_version, request)
    result = cache.get(key)
    if result is None:
        result = run_simulation(
            scenario, request.river_id, request.water_level_m, request.vehicle, request.hub_ids
        )
        cache.put(key, result)
    return key, result
