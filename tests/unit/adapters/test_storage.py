import io
import secrets

import boto3
import pytest
from botocore.response import StreamingBody
from botocore.stub import ANY, Stubber

from evaku_ai.adapters import storage
from evaku_ai.adapters.storage import R2Storage, r2_storage
from evaku_ai.core.config import Settings
from evaku_ai.core.errors import ConfigurationError, DownloadError, StorageError

BUCKET = "evaku-private"
ENDPOINT = "https://account.r2.cloudflarestorage.com"


def _client():
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=secrets.token_urlsafe(8),
        aws_secret_access_key=secrets.token_urlsafe(16),
        region_name="auto",
    )


@pytest.fixture
def stubbed():
    client = _client()
    with Stubber(client) as stub:
        yield R2Storage(client, BUCKET), stub


def _body(data: bytes) -> StreamingBody:
    return StreamingBody(io.BytesIO(data), len(data))


def test_upload_sends_type_length_and_checksum_metadata(stubbed, tmp_path):
    store, stub = stubbed
    local = tmp_path / "x.csv"
    local.write_bytes(b"abc")
    stub.add_response(
        "put_object",
        {},
        {
            "Bucket": BUCKET,
            "Key": "data/x.csv",
            "Body": ANY,
            "ContentType": "text/csv",
            "ContentLength": 3,
            "Metadata": {"sha256": "ab" * 32},
        },
    )
    store.upload(local, "data/x.csv", "text/csv", "ab" * 32)
    stub.assert_no_pending_responses()


def test_upload_failure_is_a_storage_error(stubbed, tmp_path):
    store, stub = stubbed
    local = tmp_path / "x.csv"
    local.write_bytes(b"abc")
    stub.add_client_error("put_object", service_error_code="AccessDenied", http_status_code=403)
    with pytest.raises(StorageError, match="uploading"):
        store.upload(local, "data/x.csv", "text/csv", "ab" * 32)


def test_head_reads_size_and_checksum_metadata(stubbed):
    store, stub = stubbed
    stub.add_response(
        "head_object",
        {"ContentLength": 3, "Metadata": {"sha256": "cd" * 32}},
        {"Bucket": BUCKET, "Key": "k"},
    )
    info = store.head("k")
    assert (info.key, info.size, info.sha256) == ("k", 3, "cd" * 32)


def test_head_without_metadata_has_no_checksum(stubbed):
    store, stub = stubbed
    stub.add_response("head_object", {"ContentLength": 7}, {"Bucket": BUCKET, "Key": "k"})
    assert store.head("k").sha256 is None


def test_head_of_a_missing_object_is_none(stubbed):
    store, stub = stubbed
    stub.add_client_error("head_object", service_error_code="404", http_status_code=404)
    assert store.head("k") is None


def test_head_other_errors_are_storage_errors(stubbed):
    store, stub = stubbed
    stub.add_client_error("head_object", service_error_code="403", http_status_code=403)
    with pytest.raises(StorageError, match="reading"):
        store.head("k")


def test_list_follows_pagination(stubbed):
    store, stub = stubbed
    stub.add_response(
        "list_objects_v2",
        {
            "Contents": [{"Key": "p/a", "Size": 1}],
            "IsTruncated": True,
            "NextContinuationToken": "t",
        },
        {"Bucket": BUCKET, "Prefix": "p/"},
    )
    stub.add_response(
        "list_objects_v2",
        {"Contents": [{"Key": "p/b", "Size": 2}], "IsTruncated": False},
        {"Bucket": BUCKET, "Prefix": "p/", "ContinuationToken": "t"},
    )
    found = store.list_objects("p/")
    assert [(o.key, o.size, o.sha256) for o in found] == [("p/a", 1, None), ("p/b", 2, None)]


def test_list_of_an_empty_prefix_is_empty(stubbed):
    store, stub = stubbed
    stub.add_response("list_objects_v2", {"IsTruncated": False}, {"Bucket": BUCKET, "Prefix": "p/"})
    assert store.list_objects("p/") == []


def test_list_failure_is_a_storage_error(stubbed):
    store, stub = stubbed
    stub.add_client_error(
        "list_objects_v2", service_error_code="AccessDenied", http_status_code=403
    )
    with pytest.raises(StorageError, match="listing"):
        store.list_objects("p/")


def test_download_writes_the_object(stubbed, tmp_path):
    store, stub = stubbed
    stub.add_response(
        "get_object", {"Body": _body(b"hello"), "ContentLength": 5}, {"Bucket": BUCKET, "Key": "k"}
    )
    dest = store.download("k", tmp_path / "sub" / "out.bin", max_bytes=10)
    assert dest.read_bytes() == b"hello"
    assert [p.name for p in dest.parent.iterdir()] == ["out.bin"]


def test_download_refuses_an_announced_size_over_the_limit(stubbed, tmp_path):
    store, stub = stubbed
    stub.add_response("get_object", {"Body": _body(b"0123456789"), "ContentLength": 10})
    with pytest.raises(DownloadError, match="over 5"):
        store.download("k", tmp_path / "out.bin", max_bytes=5)
    assert list(tmp_path.iterdir()) == []


def test_download_stops_when_the_body_outgrows_the_limit(stubbed, tmp_path):
    store, stub = stubbed
    stub.add_response("get_object", {"Body": _body(b"0123456789"), "ContentLength": 3})
    with pytest.raises(DownloadError, match="exceeds 5"):
        store.download("k", tmp_path / "out.bin", max_bytes=5)
    assert list(tmp_path.iterdir()) == []


def test_download_of_a_missing_object(stubbed, tmp_path):
    store, stub = stubbed
    stub.add_client_error("get_object", service_error_code="NoSuchKey", http_status_code=404)
    with pytest.raises(DownloadError, match="was not found"):
        store.download("k", tmp_path / "out.bin", max_bytes=5)


def test_download_failure_is_a_download_error(stubbed, tmp_path):
    store, stub = stubbed
    stub.add_client_error("get_object", service_error_code="InternalError", http_status_code=500)
    with pytest.raises(DownloadError, match="failed"):
        store.download("k", tmp_path / "out.bin", max_bytes=5)


def _settings(**changes):
    values = {
        "r2_endpoint": ENDPOINT,
        "r2_access_key_id": secrets.token_urlsafe(8),
        "r2_secret_access_key": secrets.token_urlsafe(16),
    }
    return Settings(_env_file=None, **(values | changes))


def test_r2_storage_builds_a_client_with_timeouts_and_retries(monkeypatch):
    seen = {}
    real = _client()

    def fake_client(service, **kwargs):
        seen.update(kwargs, service=service)
        return real

    monkeypatch.setattr(storage.boto3, "client", fake_client)
    settings = _settings()
    assert isinstance(r2_storage(settings, BUCKET), R2Storage)
    assert seen["service"] == "s3"
    assert seen["endpoint_url"] == ENDPOINT
    assert seen["region_name"] == "auto"
    assert seen["aws_access_key_id"] == settings.r2_access_key_id.get_secret_value()
    config = seen["config"]
    assert (config.connect_timeout, config.read_timeout) == (10.0, 300.0)
    assert config.retries == {"max_attempts": 5, "mode": "standard"}
    assert config.request_checksum_calculation == "when_required"
    assert config.response_checksum_validation == "when_required"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"r2_endpoint": None}, "must be set"),
        ({"r2_access_key_id": None}, "must be set"),
        ({"r2_secret_access_key": None}, "must be set"),
        ({"r2_endpoint": "http://account.r2.cloudflarestorage.com"}, "https://"),
    ],
)
def test_r2_storage_rejects_incomplete_or_insecure_settings(changes, message):
    with pytest.raises(ConfigurationError, match=message):
        r2_storage(_settings(**changes), BUCKET)
