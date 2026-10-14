from contextlib import AbstractContextManager

from evaku_ai.domain.enums import RunType
from evaku_ai.domain.models import MetricRow


def connect(dsn: str) -> AbstractContextManager[object]:
    raise NotImplementedError


def write_run(conn: object, run_id: str, run_type: RunType, results: object) -> None:
    raise NotImplementedError


def load_static_tables(conn: object, data: object) -> None:
    raise NotImplementedError


def load_metrics(conn: object, rows: list[MetricRow]) -> None:
    raise NotImplementedError
