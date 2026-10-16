import pytest

from evaku_ai.api.cache import SimulationCache
from evaku_ai.domain.enums import Vehicle
from evaku_ai.engine.flooding import depth_from_water_level, edge_status
from evaku_ai.engine.routing import plan_routes
from tests.fixtures.small_graph import NODE_LAT, NODE_LON, small_geometry, small_scenario

HUBS = {10: 0, 11: 4}
SIMULATION = {"river_id": "r1", "water_level_m": 0.0, "vehicle": "car"}
UNKNOWN_HASH = "0123456789abcdef"


@pytest.fixture
def route_client(client):
    with client:
        client.app.state.engine_data = small_scenario(hub_nodes=HUBS)
        client.app.state.road_geometry = small_geometry()
        yield client


def simulate(client, auth, **changes):
    res = client.post("/simulation", json=SIMULATION | changes, headers=auth)
    assert res.status_code == 200
    return res.json()["param_hash"]


def route(client, auth, **body):
    return client.post("/route", json={"village_id": "B"} | body, headers=auth)


def _node(i):
    return [float(NODE_LON[i]), float(NODE_LAT[i])]


def test_reachable_village_gets_the_engine_route(route_client, auth):
    key = simulate(route_client, auth)
    res = route(route_client, auth, village_id="A", param_hash=key)
    assert res.status_code == 200
    body = res.json()
    scenario = small_scenario(hub_nodes=HUBS)
    status, _ = edge_status(scenario.graph, depth_from_water_level(scenario.graph.hand_min, 0.0))
    [expected] = plan_routes(scenario, small_geometry(), status, Vehicle.CAR, [0], HUBS)
    assert body["mode"] == "land" and body["boat_point"] is None
    assert body["geojson"] == {
        "type": "LineString",
        "coordinates": [list(c) for c in expected.coordinates],
    }
    assert body["distance_m"] == pytest.approx(expected.distance_m)
    assert body["minutes"] == pytest.approx(expected.minutes)


def test_cut_village_gets_a_boat_point(route_client, auth):
    key = simulate(route_client, auth, water_level_m=1.0, hub_ids=[10])
    body = route(route_client, auth, param_hash=key).json()
    assert body["mode"] == "land_boat"
    assert body["boat_point"] == {
        "lon": NODE_LON[1],
        "lat": NODE_LAT[1],
        "water_distance_m": pytest.approx(1000.0),
    }
    assert body["distance_m"] == pytest.approx(2000.0)
    assert body["minutes"] == pytest.approx(7.0)


def test_unknown_village_is_404(route_client, auth):
    key = simulate(route_client, auth)
    res = route(route_client, auth, village_id="Z", param_hash=key)
    assert res.status_code == 404
    assert "unknown village_id" in res.json()["detail"]


def test_uncached_simulation_without_parameters_is_404(route_client, auth):
    res = route(route_client, auth, param_hash=UNKNOWN_HASH)
    assert res.status_code == 404
    assert "not cached" in res.json()["detail"]


def test_uncached_simulation_is_rebuilt_from_its_parameters(route_client, auth):
    key = simulate(route_client, auth, water_level_m=1.0)
    # Another instance: same data, empty cache.
    route_client.app.state.simulation_cache = SimulationCache(4)
    simulation = SIMULATION | {"water_level_m": 1.0}
    res = route(route_client, auth, param_hash=key, simulation=simulation)
    assert res.status_code == 200
    assert res.json()["mode"] == "land_boat"
    assert route_client.app.state.simulation_cache.get(key) is not None


def test_simulation_must_match_param_hash(route_client, auth):
    res = route(route_client, auth, param_hash=UNKNOWN_HASH, simulation=SIMULATION)
    assert res.status_code == 422
    assert "does not match" in res.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [
        {"run_id": "r1", "simulation": SIMULATION},
        {"simulation": SIMULATION},
        {"param_hash": "not-a-hash"},
        {"param_hash": UNKNOWN_HASH.upper()},
    ],
)
def test_invalid_sources_are_422(route_client, auth, body):
    assert route(route_client, auth, **body).status_code == 422


def test_published_runs_are_not_available_yet(client, auth):
    res = route(client, auth, run_id="r1")
    assert res.status_code == 501
    assert "param_hash" in res.json()["detail"]


def test_hub_id_selects_the_start(route_client, auth):
    key = simulate(route_client, auth)
    assert route(route_client, auth, param_hash=key, hub_id=99).status_code == 422
    body = route(route_client, auth, param_hash=key, hub_id=11).json()
    assert body["geojson"]["coordinates"][0] == _node(4)


def test_route_starts_from_the_simulation_hubs(route_client, auth):
    key = simulate(route_client, auth, hub_ids=[11])
    body = route(route_client, auth, param_hash=key).json()
    assert body["geojson"]["coordinates"][0] == _node(4)


def test_without_static_data_returns_503(client, auth):
    with client:
        res = route(client, auth, param_hash=UNKNOWN_HASH)
    assert res.status_code == 503
