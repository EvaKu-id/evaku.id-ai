from pathlib import Path

import yaml
from pydantic import ValidationError

from evaku_ai.core.errors import ConfigurationError
from evaku_ai.domain.region import RegionConfig

MAX_CONFIG_BYTES = 64 * 1024


def load_region(path: Path) -> RegionConfig:
    if not path.is_file():
        raise ConfigurationError(f"region config not found: {path}")
    if path.stat().st_size > MAX_CONFIG_BYTES:
        raise ConfigurationError(f"region config {path} is larger than {MAX_CONFIG_BYTES} bytes")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"region config {path} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError(f"region config {path} must be a mapping")
    try:
        region = RegionConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigurationError(f"invalid region config {path}: {_summary(exc)}") from exc
    if region.region_id != path.stem:
        raise ConfigurationError(
            f"region_id {region.region_id!r} does not match file name {path.name!r}"
        )
    return region


def _summary(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in err['loc']) or '<root>'}: {err['msg']}"
        for err in exc.errors(include_input=False)
    )
