import numpy as np
import pytest
from fastapi.testclient import TestClient

from evaku_ai.api.app import create_app
from evaku_ai.api.routers import simulation as simulation_router
from evaku_ai.core.config import Settings, get_settings
from evaku_ai.domain.enums import Vehicle
from evaku_ai.engine.simulation import run_simulation
from tests.fixtures.small_graph import small_scenario

BODY = {"river_id": "r1", "water_level_m": 2.0, "vehicle": "car"}


@pytest.fixture
def sim_client(client):
    with client:
        client.app.state.engine_data = small_scenario(hub_nodes={10: 0, 11: 4})
        yield client


def post(client, auth, **changes):
    return client.post("/simulation", json=BODY | changes, headers=auth)


def test_without_static_data_returns_503(client, auth):
    with client:
        res = post(client, auth)
    assert res.status_code == 503
    assert "not loaded" in res.json()["detail"]


def test_response_matches_engine(sim_client, auth):
    res = post(sim_client, auth)
    assert res.status_code == 200
    body = res.json()
    expected = run_simulation(
        small_scenario(hub_nodes={10: 0, 11: 4}), "r1", 2.0, Vehicle.CAR, None
    )
    assert body["data_version"] == "d1"
    assert body["roads"] == [
        {"edge_id": r.edge_id, "status": int(r.status), "depth_cm": r.depth_cm}
        for r in expected.roads
    ]
    assert body["villages"] == [
        {
            "village_id": v.village_id,
            "status": v.status.value,
            "isolated_population": v.isolated_population,
            "priority_score": v.priority_score,
        }
        for v in expected.villages
    ]
    assert body["summary"] == {
        "isolated_villages": expected.summary.isolated_villages,
        "isolated_population": expected.summary.isolated_population,
        "cut_road_km": expected.summary.cut_road_km,
    }
    assert isinstance(body["ms"], int) and body["ms"] >= 0


def test_two_metres_isolates_village_b(sim_client, auth):
    body = post(sim_client, auth).json()
    assert body["villages"][1]["status"] == "fully_isolated"
    assert body["summary"]["isolated_population"] == 200


def test_cut_road_km_depends_on_vehicle(sim_client, auth):
    car = post(sim_client, auth, water_level_m=1.0).json()["summary"]["cut_road_km"]
    truck = post(sim_client, auth, water_level_m=1.0, vehicle="truck").json()
    assert car == pytest.approx(1.5)
    assert truck["summary"]["cut_road_km"] == 0.0


def test_param_hash_identifies_the_scenario(sim_client, auth):
    first = post(sim_client, auth, hub_ids=[10, 11]).json()["param_hash"]
    assert post(sim_client, auth, hub_ids=[11, 10, 10]).json()["param_hash"] == first
    assert post(sim_client, auth, hub_ids=[10]).json()["param_hash"] != first
    assert (
        post(sim_client, auth).json()["param_hash"]
        == post(sim_client, auth, hub_ids=[]).json()["param_hash"]
    )
    assert (
        post(sim_client, auth, water_level_m=1.0).json()["param_hash"]
        != post(sim_client, auth).json()["param_hash"]
    )


def test_repeated_request_uses_cache(sim_client, auth, monkeypatch):
    calls = []

    def counting(*args, **kwargs):
        calls.append(args)
        return run_simulation(*args, **kwargs)

    monkeypatch.setattr(simulation_router, "run_simulation", counting)
    first = post(sim_client, auth, water_level_m=1.5).json()
    second = post(sim_client, auth, water_level_m=1.5).json()
    assert len(calls) == 1
    assert {**first, "ms": 0} == {**second, "ms": 0}


@pytest.mark.parametrize(
    ("changes", "unknown"), [({"river_id": "r9"}, "r9"), ({"hub_ids": [99]}, "99")]
)
def test_unknown_ids_return_422(sim_client, auth, changes, unknown):
    res = post(sim_client, auth, **changes)
    assert res.status_code == 422
    assert unknown in res.json()["detail"]


def test_synthetic_data_is_served_when_enabled(settings, auth):
    app = create_app()
    synthetic = settings.model_copy(update={"engine_data": "synthetic"})
    app.dependency_overrides[get_settings] = lambda: synthetic
    with TestClient(app) as client:
        res = post(client, auth, river_id="synthetic-river", water_level_m=1.5)
    assert res.status_code == 200
    body = res.json()
    assert body["data_version"] == "synthetic"
    assert body["summary"]["isolated_villages"] > 0
    assert np.isfinite(body["summary"]["cut_road_km"])


def test_production_rejects_synthetic_data():
    with pytest.raises(ValueError, match="synthetic"):
        Settings(_env_file=None, app_env="production", engine_data="synthetic")
