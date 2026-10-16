import hashlib
import uuid

import pytest

from evaku_ai.adapters.storage import r2_storage
from evaku_ai.core.config import Settings

pytestmark = pytest.mark.integration


@pytest.fixture
def storage():
    settings = Settings()
    if not settings.r2_configured:
        pytest.skip("R2_ENDPOINT, R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY are not set")
    return r2_storage(settings, settings.r2_bucket_private), settings


def test_upload_head_download_and_cleanup(storage, tmp_path):
    store, settings = storage
    key = f"test/evaku-ai-integration/{uuid.uuid4()}/x.txt"
    payload = b"evaku integration check"
    sha = hashlib.sha256(payload).hexdigest()
    local = tmp_path / "x.txt"
    local.write_bytes(payload)
    try:
        store.upload(local, key, "text/csv", sha)
        info = store.head(key)
        assert info is not None and (info.size, info.sha256) == (len(payload), sha)
        assert [o.key for o in store.list_objects(key.rsplit("/", 1)[0] + "/")] == [key]
        assert store.download(key, tmp_path / "back.txt", 1024).read_bytes() == payload
    finally:
        store._client.delete_object(Bucket=settings.r2_bucket_private, Key=key)
    assert store.head(key) is None
