import pytest

from evaku_ai.core.config import Settings, get_settings

SIMULATION = {"river_id": "main-river", "water_level_m": 1.2, "vehicle": "car"}


@pytest.mark.parametrize("headers", [{}, {"X-Model-Key": "wrong"}])
def test_protected_endpoints_reject_missing_or_wrong_key(client, headers):
    res = client.post("/simulation", json=SIMULATION, headers=headers)
    assert res.status_code == 401


def test_valid_key_passes_auth(client, auth):
    res = client.post("/simulation", json=SIMULATION, headers=auth)
    assert res.status_code == 503


def test_unconfigured_key_returns_503(client):
    client.app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None)
    res = client.post("/simulation", json=SIMULATION, headers={"X-Model-Key": "x"})
    assert res.status_code == 503


def test_validation_runs_before_stub(client, auth):
    res = client.post("/simulation", json={**SIMULATION, "water_level_m": 9}, headers=auth)
    assert res.status_code == 422


def test_recalc_requires_at_least_one_correction(client, auth):
    body = {"run_id": "r1", "corrections": []}
    assert client.post("/recalc", json=body, headers=auth).status_code == 422
    body["corrections"] = [{"edge_id": 3, "status": 2}]
    assert client.post("/recalc", json=body, headers=auth).status_code == 501


def test_route_requires_exactly_one_source(client, auth):
    body = {"village_id": "v1", "vehicle": "truck"}
    assert client.post("/route", json=body, headers=auth).status_code == 422
    both = {**body, "run_id": "r1", "param_hash": "h1"}
    assert client.post("/route", json=both, headers=auth).status_code == 422
    one = {**body, "run_id": "r1"}
    assert client.post("/route", json=one, headers=auth).status_code == 501
