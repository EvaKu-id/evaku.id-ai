from evaku_ai.domain.arrays import BoolArray, FloatArray
from evaku_ai.inference.registry import ModelPackage


def predict_water(
    package: ModelPackage, image: FloatArray, tile: int = 512, overlap: int = 64
) -> BoolArray:
    raise NotImplementedError
