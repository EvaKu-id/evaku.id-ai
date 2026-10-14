from scipy import sparse

from evaku_ai.domain.arrays import FloatArray, IntArray
from evaku_ai.domain.models import VillageResult


def travel_time_from_hubs(
    csr: sparse.csr_matrix, hub_nodes: IntArray
) -> tuple[FloatArray, IntArray, IntArray]:
    raise NotImplementedError


def classify_villages(
    time_flood_s: FloatArray,
    time_normal_s: FloatArray,
    pop_node: FloatArray,
    pop_village: FloatArray,
    flooded_pop: FloatArray,
    village_ids: list[str],
) -> list[VillageResult]:
    raise NotImplementedError
