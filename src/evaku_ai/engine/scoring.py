from evaku_ai.domain.arrays import FloatArray
from evaku_ai.domain.rules import ScoreWeights


def priority_score(
    components: dict[str, FloatArray], weights: ScoreWeights = ScoreWeights()
) -> FloatArray:
    raise NotImplementedError
