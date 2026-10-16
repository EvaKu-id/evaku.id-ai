import secrets

import pytest

from evaku_ai.core.config import Settings

R2 = {
    "r2_endpoint": "https://account.r2.cloudflarestorage.com",
    "r2_access_key_id": secrets.token_urlsafe(8),
    "r2_secret_access_key": secrets.token_urlsafe(16),
}


def test_r2_is_configured_only_with_endpoint_and_both_keys():
    assert Settings(_env_file=None, **R2).r2_configured is True
    assert Settings(_env_file=None).r2_configured is False


@pytest.mark.parametrize("missing", list(R2))
def test_r2_needs_every_setting(missing):
    partial = {k: v for k, v in R2.items() if k != missing}
    assert Settings(_env_file=None, **partial).r2_configured is False
