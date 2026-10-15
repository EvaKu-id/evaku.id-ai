import hashlib

import httpx
import pytest

from evaku_ai.adapters.download import fetch
from evaku_ai.core.errors import ChecksumMismatchError, DownloadError
from evaku_ai.domain.region import RemoteFile

BODY = b"boundary bytes " * 1000
DIGEST = hashlib.sha256(BODY).hexdigest()
URL = "https://example.org/boundaries.gdb.zip"


def remote(tmp_path, sha256=DIGEST):
    path = tmp_path / "data" / "raw" / "boundaries.gdb.zip"
    rf = RemoteFile(
        path="data/raw/boundaries.gdb.zip",
        url=URL,
        sha256=sha256,
        source="test",
        license="CC BY-IGO",
    )
    return rf, path


def client(calls, status=200, body=BODY):
    def handler(request):
        calls.append(request.url)
        return httpx.Response(status, content=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_download_verifies_and_saves(tmp_path):
    rf, path = remote(tmp_path)
    calls = []
    assert fetch(rf, client=client(calls)).read_bytes() == BODY
    assert calls == [httpx.URL(URL)]
    assert not path.with_name(path.name + ".part").exists()


def test_existing_valid_file_skips_network(tmp_path):
    rf, path = remote(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(BODY)
    calls = []
    fetch(rf, client=client(calls))
    assert calls == []


def test_existing_corrupt_file_is_downloaded_again(tmp_path):
    rf, path = remote(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"corrupt")
    calls = []
    assert fetch(rf, client=client(calls)).read_bytes() == BODY
    assert len(calls) == 1


def test_checksum_mismatch_leaves_no_file(tmp_path):
    rf, path = remote(tmp_path, sha256="0" * 64)
    with pytest.raises(ChecksumMismatchError, match="expected"):
        fetch(rf, client=client([]))
    assert not path.exists()
    assert not path.with_name(path.name + ".part").exists()


def test_http_error_raises(tmp_path):
    rf, _ = remote(tmp_path)
    with pytest.raises(DownloadError, match="HTTP 404"):
        fetch(rf, client=client([], status=404))


def test_size_limit(tmp_path):
    rf, path = remote(tmp_path)
    with pytest.raises(DownloadError, match="exceeds"):
        fetch(rf, client=client([]), max_bytes=100)
    assert not path.exists()


def test_transport_error_raises(tmp_path):
    rf, _ = remote(tmp_path)

    def broken(request):
        raise httpx.ConnectError("offline", request=request)

    with pytest.raises(DownloadError, match="offline"):
        fetch(rf, client=httpx.Client(transport=httpx.MockTransport(broken)))
