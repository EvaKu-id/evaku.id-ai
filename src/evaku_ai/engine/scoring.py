from dataclasses import dataclass, fields

import numpy as np

from evaku_ai.core.errors import InvalidInputError
from evaku_ai.domain.arrays import BoolArray, FloatArray
from evaku_ai.domain.rules import ScoreWeights

COMPONENTS = tuple(field.name for field in fields(ScoreWeights))
SCORE_MAX = 100.0


@dataclass(frozen=True)
class ScoreResult:
    score: FloatArray
    raw: dict[str, FloatArray]
    value: dict[str, FloatArray]
    weight: dict[str, float]

    def breakdown(self, village: int) -> dict[str, dict[str, float]]:
        return {
            name: {
                "raw": float(self.raw[name][village]),
                "value": float(self.value[name][village]),
                "weight": self.weight[name],
            }
            for name in COMPONENTS
        }


def priority_scores(
    components: dict[str, FloatArray],
    eligible: BoolArray,
    weights: ScoreWeights = ScoreWeights(),
) -> ScoreResult:
    _validate(components, eligible)
    # Components without data hand their weight to the others, so scores still span 0-100.
    given = {name: getattr(weights, name) if name in components else 0.0 for name in COMPONENTS}
    total_weight = sum(given.values())
    if total_weight <= 0:
        raise InvalidInputError("the given components carry no weight")
    weight = {name: float(w / total_weight) for name, w in given.items()}

    empty = np.zeros(eligible.shape[0])
    raw = {name: np.asarray(components.get(name, empty), dtype=np.float64) for name in COMPONENTS}
    value = {name: _normalize(raw[name], eligible) for name in COMPONENTS}
    score = SCORE_MAX * np.sum([weight[name] * value[name] for name in COMPONENTS], axis=0)
    return ScoreResult(score=score, raw=raw, value=value, weight=weight)


def _validate(components: dict[str, FloatArray], eligible: BoolArray) -> None:
    if eligible.dtype != np.bool_ or eligible.ndim != 1:
        raise InvalidInputError("eligible must be a 1-D bool array")
    if not components:
        raise InvalidInputError("at least one score component is required")
    unknown = set(components) - set(COMPONENTS)
    if unknown:
        raise InvalidInputError(f"unknown score components: {sorted(unknown)}")
    for name, values in components.items():
        if np.shape(values) != eligible.shape:
            raise InvalidInputError(f"{name} must have length {eligible.shape[0]}")
        if not np.all(np.isfinite(values) & (np.asarray(values) >= 0)):
            raise InvalidInputError(f"{name} must be finite and >= 0")


def _normalize(raw: FloatArray, eligible: BoolArray) -> FloatArray:
    value = np.zeros_like(raw)
    if not eligible.any():
        return value
    subset = raw[eligible]
    low, high = subset.min(), subset.max()
    # With no spread (e.g. a single isolated village) any non-zero need counts in full.
    value[eligible] = (subset - low) / (high - low) if high > low else (subset > 0).astype(float)
    return value
