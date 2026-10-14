from pathlib import Path


def upload(local: Path, bucket: str, key: str, content_type: str | None = None) -> None:
    raise NotImplementedError


def download_prefix(bucket: str, prefix: str, dest: Path) -> list[Path]:
    raise NotImplementedError
