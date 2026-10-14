from datetime import datetime


def init(service_account: str, key_file: str) -> None:
    raise NotImplementedError


def latest_sentinel1(bbox: tuple[float, float, float, float], since: datetime) -> list[str]:
    raise NotImplementedError
