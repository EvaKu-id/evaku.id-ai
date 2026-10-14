from evaku_ai.domain.arrays import BoolArray, FloatArray, IntArray


def path_from_predecessors(predecessors: IntArray, target: int) -> list[int]:
    raise NotImplementedError


def boat_point(
    reachable: BoolArray, node_x: FloatArray, node_y: FloatArray, target: int
) -> tuple[int, float]:
    raise NotImplementedError
