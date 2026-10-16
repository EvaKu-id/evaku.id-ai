import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol, cast

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from evaku_ai.core.config import Settings
from evaku_ai.core.errors import ConfigurationError, DownloadError, StorageError

log = logging.getLogger(__name__)

CONNECT_TIMEOUT_S = 10.0
READ_TIMEOUT_S = 300.0
MAX_ATTEMPTS = 5
CHUNK_BYTES = 1 << 20
SHA256_META = "sha256"
MISSING_CODES = frozenset({"404", "NoSuchKey", "NotFound"})


@dataclass(frozen=True)
class ObjectInfo:
    key: str
    size: int
    sha256: str | None


class Storage(Protocol):
    def list_objects(self, prefix: str) -> list[ObjectInfo]: ...

    def head(self, key: str) -> ObjectInfo | None: ...

    def upload(
        self,
        local: Path,
        key: str,
        content_type: str,
        sha256: str,
        cache_control: str | None = None,
    ) -> None: ...

    def download(self, key: str, dest: Path, max_bytes: int) -> Path: ...


class _Body(Protocol):
    def read(self, amt: int = ...) -> bytes: ...

    def close(self) -> None: ...


class _Paginator(Protocol):
    def paginate(self, *, Bucket: str, Prefix: str) -> Iterable[Mapping[str, object]]: ...


class S3Client(Protocol):
    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: BinaryIO,
        ContentType: str,
        ContentLength: int,
        Metadata: Mapping[str, str],
        CacheControl: str = ...,
    ) -> Mapping[str, object]: ...

    def get_object(self, *, Bucket: str, Key: str) -> Mapping[str, object]: ...

    def head_object(self, *, Bucket: str, Key: str) -> Mapping[str, object]: ...

    def get_paginator(self, operation_name: str) -> _Paginator: ...


class R2Storage:
    def __init__(self, client: S3Client, bucket: str) -> None:
        self._client = client
        self._bucket = bucket

    def list_objects(self, prefix: str) -> list[ObjectInfo]:
        found: list[ObjectInfo] = []
        try:
            pages = self._client.get_paginator("list_objects_v2").paginate(
                Bucket=self._bucket, Prefix=prefix
            )
            for page in pages:
                for item in _contents(page):
                    found.append(ObjectInfo(str(item["Key"]), _as_int(item.get("Size")), None))
        except (ClientError, BotoCoreError) as exc:
            raise StorageError(f"listing {self._bucket}/{prefix} failed: {exc}") from exc
        return found

    def head(self, key: str) -> ObjectInfo | None:
        try:
            response = self._client.head_object(Bucket=self._bucket, Key=key)
        except ClientError as exc:
            if _error_code(exc) in MISSING_CODES:
                return None
            raise StorageError(f"reading {self._bucket}/{key} failed: {exc}") from exc
        except BotoCoreError as exc:
            raise StorageError(f"reading {self._bucket}/{key} failed: {exc}") from exc
        return ObjectInfo(key, _as_int(response.get("ContentLength")), _metadata_sha(response))

    def upload(
        self,
        local: Path,
        key: str,
        content_type: str,
        sha256: str,
        cache_control: str | None = None,
    ) -> None:
        size = local.stat().st_size
        try:
            with local.open("rb") as body:
                if cache_control is None:
                    self._client.put_object(
                        Bucket=self._bucket,
                        Key=key,
                        Body=body,
                        ContentType=content_type,
                        ContentLength=size,
                        Metadata={SHA256_META: sha256},
                    )
                else:
                    self._client.put_object(
                        Bucket=self._bucket,
                        Key=key,
                        Body=body,
                        ContentType=content_type,
                        ContentLength=size,
                        Metadata={SHA256_META: sha256},
                        CacheControl=cache_control,
                    )
        except (ClientError, BotoCoreError) as exc:
            raise StorageError(f"uploading {self._bucket}/{key} failed: {exc}") from exc
        log.info("uploaded %s/%s (%.1f MB)", self._bucket, key, size / 1e6)

    def download(self, key: str, dest: Path, max_bytes: int) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        body: _Body | None = None
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
            announced = _as_int(response.get("ContentLength"))
            if announced > max_bytes:
                raise DownloadError(f"{self._bucket}/{key} is {announced} bytes, over {max_bytes}")
            body = cast(_Body, response["Body"])
            size = 0
            with part.open("wb") as out:
                while chunk := body.read(CHUNK_BYTES):
                    size += len(chunk)
                    if size > max_bytes:
                        raise DownloadError(f"{self._bucket}/{key} exceeds {max_bytes} bytes")
                    out.write(chunk)
            part.replace(dest)
        except ClientError as exc:
            reason = "was not found" if _error_code(exc) in MISSING_CODES else f"failed: {exc}"
            raise DownloadError(f"download of {self._bucket}/{key} {reason}") from exc
        except BotoCoreError as exc:
            raise DownloadError(f"download of {self._bucket}/{key} failed: {exc}") from exc
        finally:
            if body is not None:
                body.close()
            part.unlink(missing_ok=True)
        return dest


def r2_storage(settings: Settings, bucket: str) -> R2Storage:
    if not settings.r2_configured:
        raise ConfigurationError(
            "R2_ENDPOINT, R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY must be set to use R2"
        )
    endpoint = settings.r2_endpoint or ""
    if not endpoint.startswith("https://"):
        raise ConfigurationError("R2_ENDPOINT must start with https://")
    key_id, secret = settings.r2_access_key_id, settings.r2_secret_access_key
    if key_id is None or secret is None:
        raise ConfigurationError("R2 credentials are missing")
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key_id.get_secret_value(),
        aws_secret_access_key=secret.get_secret_value(),
        region_name="auto",
        config=Config(
            connect_timeout=CONNECT_TIMEOUT_S,
            read_timeout=READ_TIMEOUT_S,
            retries={"max_attempts": MAX_ATTEMPTS, "mode": "standard"},
            # botocore 1.36+ adds CRC32 trailers that R2 may reject; our SHA-256 covers integrity.
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )
    return R2Storage(client, bucket)


def _contents(page: Mapping[str, object]) -> list[Mapping[str, object]]:
    contents = page.get("Contents")
    if contents is None:
        return []
    if not isinstance(contents, list):
        raise StorageError("unexpected listing response from R2")
    return [item for item in contents if isinstance(item, Mapping)]


def _as_int(value: object) -> int:
    return value if isinstance(value, int) else 0


def _metadata_sha(response: Mapping[str, object]) -> str | None:
    metadata = response.get("Metadata")
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get(SHA256_META)
    return value if isinstance(value, str) else None


def _error_code(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if isinstance(response, Mapping):
        error = response.get("Error")
        if isinstance(error, Mapping):
            return str(error.get("Code", ""))
    return ""
