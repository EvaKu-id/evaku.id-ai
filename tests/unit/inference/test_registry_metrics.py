import json

import pytest

from evaku_ai.core.errors import InvalidModelPackageError
from evaku_ai.domain.models import MetricRow
from evaku_ai.inference.registry import METRICS_FILE, ModelPackage, parse_metrics, read_metrics

VALID = [
    {"model": "unet", "name": "iou", "value": 0.71, "note": "Sentinel-1 test scenes"},
    {"model": "lgbm", "name": "pr_auc", "value": 0.64},
]


def test_parse_metrics_returns_rows():
    rows = parse_metrics(json.dumps(VALID).encode())
    assert rows == [
        MetricRow("unet", "iou", 0.71, "Sentinel-1 test scenes"),
        MetricRow("lgbm", "pr_auc", 0.64, None),
    ]


def test_parse_metrics_accepts_empty_list():
    assert parse_metrics(b"[]") == []


@pytest.mark.parametrize(
    "raw",
    [
        b"not json",
        b'{"model": "unet"}',
        json.dumps([{"model": "unet", "name": "iou"}]).encode(),
        json.dumps([{"model": "", "name": "iou", "value": 1}]).encode(),
        json.dumps([{**VALID[0], "extra": 1}]).encode(),
        b'[{"model": "unet", "name": "iou", "value": NaN}]',
    ],
)
def test_parse_metrics_rejects_malformed(raw):
    with pytest.raises(InvalidModelPackageError, match=METRICS_FILE):
        parse_metrics(raw)


def test_read_metrics_from_package(tmp_path):
    (tmp_path / METRICS_FILE).write_text(json.dumps(VALID))
    rows = read_metrics(ModelPackage("v1", tmp_path))
    assert [r.name for r in rows] == ["iou", "pr_auc"]


def test_read_metrics_missing_file(tmp_path):
    with pytest.raises(InvalidModelPackageError, match="missing"):
        read_metrics(ModelPackage("v1", tmp_path))
