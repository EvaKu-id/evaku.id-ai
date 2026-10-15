import numpy as np

from evaku_ai.engine.evacuation import EvacuationSites


def sample_sites() -> EvacuationSites:
    return EvacuationSites(
        facility_ids=np.array([7, 9], dtype=np.int64),
        nodes=np.array([0, 5], dtype=np.int64),
        hand=np.array([3.5, 0.0]),
        types=["school", "hospital"],
        preferred=np.array([True, False]),
    )
