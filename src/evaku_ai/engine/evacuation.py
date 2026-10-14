from scipy import sparse

from evaku_ai.domain.arrays import BoolArray, FloatArray, IntArray


def safe_sites(facility_hand: FloatArray, water_level_m: float, margin_m: float = 1.0) -> BoolArray:
    raise NotImplementedError


def nearest_safe_sites(
    csr: sparse.csr_matrix,
    village_node: int,
    site_nodes: IntArray,
    k: int = 3,
    limit_s: float | None = None,
) -> list[tuple[int, float]]:
    raise NotImplementedError
