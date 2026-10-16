from pathlib import Path

import pytest
import yaml

from evaku_ai.adapters.release import RELEASE_VERSION_PATTERN
from evaku_ai.core.config import Settings

WORKFLOW = Path(".github/workflows/deploy-model-api.yml")
DEPLOY_ACTION = "google-github-actions/deploy-cloudrun"
REQUIRED_FLAGS = (
    "--service-account=${{ env.RUNTIME_SERVICE_ACCOUNT }}",
    "--concurrency=4",
    "--min-instances=0",
    "--max-instances=2",
    "--memory=2Gi",
    "--cpu=2",
    "--cpu-boost",
    "--allow-unauthenticated",
)
EXPECTED_ENV_VARS = [
    "APP_ENV=production",
    "DATA_DIR=${{ env.DATA_DIR }}",
    "DATA_VERSION=${{ vars.DATA_VERSION }}",
    "MODEL_VERSION=${{ env.MODEL_VERSION }}",
]
EXPECTED_SECRETS = [
    "MODEL_API_KEY=model-api-key:latest",
    "R2_ENDPOINT=r2-endpoint:latest",
    "R2_ACCESS_KEY_ID=r2-access-key-id:latest",
    "R2_SECRET_ACCESS_KEY=r2-secret-access-key:latest",
]


@pytest.fixture(scope="module")
def workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def steps(workflow):
    return workflow["jobs"]["deploy"]["steps"]


def _triggers(workflow):
    # PyYAML follows YAML 1.1 and reads the bare key `on` as the boolean True.
    return workflow.get("on", workflow.get(True))


def _index(steps, *, uses=None, name=None, step_id=None):
    for position, step in enumerate(steps):
        if uses is not None and str(step.get("uses", "")).startswith(uses):
            return position
        if name is not None and step.get("name") == name:
            return position
        if step_id is not None and step.get("id") == step_id:
            return position
    pytest.fail(f"no step with uses={uses} name={name} id={step_id}")


def _lines(block):
    return [line.strip() for line in block.splitlines() if line.strip()]


def test_triggers_are_manual_run_and_release_tag_only(workflow):
    triggers = _triggers(workflow)
    assert "workflow_dispatch" in triggers
    assert triggers["push"] == {"tags": ["model-api-v*"]}
    assert set(triggers) == {"workflow_dispatch", "push"}


def test_deploy_is_limited_to_main_and_release_tags(workflow):
    condition = workflow["jobs"]["deploy"]["if"]
    assert "refs/heads/main" in condition
    assert "refs/tags/model-api-v" in condition


def test_permissions_are_minimal(workflow):
    assert workflow["permissions"] == {"contents": "read", "id-token": "write"}


def test_deploys_never_run_in_parallel_or_cancel_each_other(workflow):
    assert workflow["concurrency"] == {"group": "deploy-model-api", "cancel-in-progress": False}


def test_login_uses_workload_identity_without_key_files(steps):
    auth = steps[_index(steps, uses="google-github-actions/auth")]["with"]
    assert auth == {
        "workload_identity_provider": "${{ vars.GCP_WIF_PROVIDER }}",
        "service_account": "${{ vars.GCP_DEPLOY_SERVICE_ACCOUNT }}",
    }
    text = WORKFLOW.read_text(encoding="utf-8")
    for forbidden in ("credentials_json", "GOOGLE_APPLICATION_CREDENTIALS", "secrets."):
        assert forbidden not in text


def test_deploy_flags(workflow, steps):
    flags = " ".join(steps[_index(steps, uses=DEPLOY_ACTION)]["with"]["flags"].split())
    for required in REQUIRED_FLAGS:
        assert required in flags
    assert workflow["env"]["RUNTIME_SERVICE_ACCOUNT"].startswith("evaku-model-api@")


def test_env_vars_are_exact_and_overwrite_the_service(workflow, steps):
    options = steps[_index(steps, uses=DEPLOY_ACTION)]["with"]
    assert options["env_vars_update_strategy"] == "overwrite"
    assert _lines(options["env_vars"]) == EXPECTED_ENV_VARS
    assert workflow["env"]["DATA_DIR"] == "/tmp/evaku-data"  # noqa: S108
    assert "'dev'" in workflow["env"]["MODEL_VERSION"]


def test_secrets_are_exact_references_and_overwrite_the_service(steps):
    options = steps[_index(steps, uses=DEPLOY_ACTION)]["with"]
    assert options["secrets_update_strategy"] == "overwrite"
    assert _lines(options["secrets"]) == EXPECTED_SECRETS


def test_image_is_the_api_target_tagged_with_the_commit(steps):
    build = steps[_index(steps, name="Build and push image")]["run"]
    assert "--target api" in build
    assert "${{ github.sha }}" in build
    image = steps[_index(steps, uses=DEPLOY_ACTION)]["with"]["image"]
    assert image.endswith(":${{ github.sha }}")


def test_variables_are_checked_before_anything_else(steps):
    first = steps[0]
    assert first["name"] == "Check required variables"
    assert _index(steps, name="Check required variables") < _index(steps, uses="actions/checkout")
    assert RELEASE_VERSION_PATTERN.replace(r"\d", "[0-9]") in first["run"]


def test_smoke_test_follows_deploy_and_summary_is_last(steps):
    smoke = _index(steps, name="Smoke test")
    assert smoke > _index(steps, step_id="deploy")
    for expected in ("/health", "data_version", "401"):
        assert expected in steps[smoke]["run"]
    assert _index(steps, name="Summary") == len(steps) - 1


def test_region_and_registry_path(workflow):
    assert workflow["env"]["REGION"] == "asia-southeast1"
    assert workflow["env"]["IMAGE"].endswith("/evaku/model-api")


def test_configured_names_exist_in_settings(steps):
    options = steps[_index(steps, uses=DEPLOY_ACTION)]["with"]
    names = [
        line.split("=", 1)[0] for line in _lines(options["env_vars"]) + _lines(options["secrets"])
    ]
    assert names
    for name in names:
        assert name.lower() in Settings.model_fields
