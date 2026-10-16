from pathlib import Path

from evaku_ai.adapters.storage import ObjectInfo
from evaku_ai.core.errors import DownloadError, StorageError


class FakeStorage:
    def __init__(self, fail_on: tuple[str, ...] = ()) -> None:
        self.objects: dict[str, tuple[bytes, str, str]] = {}
        self.calls: list[tuple[str, str]] = []
        self.cache_control: dict[str, str | None] = {}
        self.fail_on = set(fail_on)

    def list_objects(self, prefix: str) -> list[ObjectInfo]:
        self.calls.append(("list", prefix))
        return [
            ObjectInfo(key, len(data), None)
            for key, (data, _, _) in sorted(self.objects.items())
            if key.startswith(prefix)
        ]

    def head(self, key: str) -> ObjectInfo | None:
        self.calls.append(("head", key))
        if key not in self.objects:
            return None
        data, _, sha = self.objects[key]
        return ObjectInfo(key, len(data), sha)

    def upload(
        self,
        local: Path,
        key: str,
        content_type: str,
        sha256: str,
        cache_control: str | None = None,
    ) -> None:
        self.calls.append(("upload", key))
        if key in self.fail_on:
            raise StorageError(f"upload of {key} failed")
        self.objects[key] = (local.read_bytes(), content_type, sha256)
        self.cache_control[key] = cache_control

    def download(self, key: str, dest: Path, max_bytes: int) -> Path:
        self.calls.append(("download", key))
        if key in self.fail_on or key not in self.objects:
            raise DownloadError(f"download of {key} failed")
        data = self.objects[key][0]
        if len(data) > max_bytes:
            raise DownloadError(f"{key} is {len(data)} bytes, over {max_bytes}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return dest
