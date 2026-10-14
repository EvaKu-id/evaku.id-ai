from __future__ import annotations

import pandas as pd

from evaku_ai.inference.registry import ModelPackage


def predict_probability(package: ModelPackage, features: pd.DataFrame) -> pd.Series[float]:
    raise NotImplementedError


def water_level_from_discharge(package: ModelPackage, river_id: str, discharge_m3s: float) -> float:
    raise NotImplementedError
