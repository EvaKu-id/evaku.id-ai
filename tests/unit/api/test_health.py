from evaku_ai import __version__


def test_health_is_public_and_reports_versions(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {
        "status": "ok",
        "version": __version__,
        "data_version": "d1",
        "model_version": "m1",
    }
