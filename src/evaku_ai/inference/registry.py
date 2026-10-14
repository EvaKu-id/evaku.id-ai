from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from evaku_ai.core.errors import InvalidModelPackageError
from evaku_ai.domain.models import MetricRow

PACKAGE_FILES = (
    "unet.onnx",
    "lgbm.txt",
    "kalibrasi_peluang.json",
    "kalibrasi_debit.json",
    "fitur.json",
    "metrics.json",
    "manifest.json",
)
METRICS_FILE = "metrics.json"


@dataclass(frozen=True)
class ModelPackage:
    version: str
    root: Path

    def path(self, name: str) -> Path:
        return self.root / name


class _MetricEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=64)
    value: float = Field(allow_inf_nan=False)
    note: str | None = Field(default=None, max_length=500)


_METRICS = TypeAdapter(list[_MetricEntry])


def parse_metrics(raw: bytes) -> list[MetricRow]:
    try:
        entries = _METRICS.validate_json(raw)
    except ValidationError as exc:
        raise InvalidModelPackageError(f"{METRICS_FILE} is malformed: {exc}") from exc
    return [MetricRow(e.model, e.name, e.value, e.note) for e in entries]


def read_metrics(package: ModelPackage) -> list[MetricRow]:
    path = package.path(METRICS_FILE)
    if not path.is_file():
        raise InvalidModelPackageError(f"{METRICS_FILE} missing in model {package.version}")
    return parse_metrics(path.read_bytes())


def fetch(version: str, artifacts_dir: Path) -> ModelPackage:
    raise NotImplementedError
