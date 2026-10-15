import hashlib
import logging
from pathlib import Path

import httpx

from evaku_ai.core.errors import ChecksumMismatchError, DownloadError
from evaku_ai.domain.region import RemoteFile

log = logging.getLogger(__name__)

MAX_DOWNLOAD_BYTES = 1_000_000_000
CHUNK_BYTES = 1 << 20
TIMEOUT = httpx.Timeout(30.0, read=300.0)


def fetch(
    remote: RemoteFile,
    *,
    client: httpx.Client | None = None,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
) -> Path:
    path = remote.path
    if path.is_file() and file_sha256(path) == remote.sha256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    log.info("downloading %s to %s", remote.url, path)
    try:
        digest = _stream(str(remote.url), part, client, max_bytes)
        if digest != remote.sha256:
            raise ChecksumMismatchError(
                f"sha256 mismatch for {remote.url}: expected {remote.sha256}, got {digest}"
            )
        part.replace(path)
    finally:
        part.unlink(missing_ok=True)
    return path


def _stream(url: str, part: Path, client: httpx.Client | None, max_bytes: int) -> str:
    owned = client is None
    http = client or httpx.Client(timeout=TIMEOUT, follow_redirects=True)
    digest = hashlib.sha256()
    size = 0
    try:
        with http.stream("GET", url) as response:
            if response.status_code >= 400:
                raise DownloadError(f"download of {url} failed with HTTP {response.status_code}")
            with part.open("wb") as out:
                for chunk in response.iter_bytes(CHUNK_BYTES):
                    size += len(chunk)
                    if size > max_bytes:
                        raise DownloadError(f"download of {url} exceeds {max_bytes} bytes")
                    digest.update(chunk)
                    out.write(chunk)
    except httpx.HTTPError as exc:
        raise DownloadError(f"download of {url} failed: {exc}") from exc
    finally:
        if owned:
            http.close()
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()
