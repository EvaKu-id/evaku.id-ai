import numpy as np
import pytest

from evaku_ai.core.errors import InvalidInputError
from evaku_ai.engine.bundle import KEYS, bundle_from_arrays, bundle_to_arrays
from evaku_ai.engine.synthetic import synthetic_scenario
from tests.fixtures.evacuation import sample_sites


def _assert_same(left, right):
    assert left.keys() == right.keys()
    for key in left:
        np.testing.assert_array_equal(left[key], right[key], err_msg=key)


def test_round_trip_keeps_every_array():
    arrays = bundle_to_arrays(synthetic_scenario(), sample_sites())
    assert set(arrays) == KEYS
    rebuilt, sites = bundle_from_arrays(arrays)
    _assert_same(bundle_to_arrays(rebuilt, sites), arrays)
    assert sites.types == ["school", "hospital"]
    assert rebuilt.hub_nodes == synthetic_scenario().hub_nodes
    assert rebuilt.rivers == synthetic_scenario().rivers


def test_narrow_integer_types_are_widened():
    arrays = dict(bundle_to_arrays(synthetic_scenario(), sample_sites()))
    arrays["edge_u"] = arrays["edge_u"].astype(np.int32)
    arrays["edge_v"] = arrays["edge_v"].astype(np.int32)
    assert bundle_from_arrays(arrays)[0].graph.edge_u.dtype == np.int64


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda a: a.pop("hand_min"), "missing \\['hand_min'\\]"),
        (lambda a: a.__setitem__("surprise", np.zeros(1)), "extra \\['surprise'\\]"),
        (lambda a: a.__setitem__("edge_u", a["edge_u"].astype(float)), "edge_u must hold integers"),
        (
            lambda a: a.__setitem__("is_bridge", a["is_bridge"].astype(int)),
            "is_bridge must hold booleans",
        ),
        (lambda a: a.__setitem__("rivers", np.array([1])), "rivers must hold text"),
        (lambda a: a.__setitem__("hub_ids", a["hub_ids"][:1]), "same length"),
        (lambda a: a.__setitem__("site_preferred", np.array([1, 0])), "site_preferred"),
    ],
)
def test_bad_arrays_are_rejected(change, message):
    arrays = dict(bundle_to_arrays(synthetic_scenario(), sample_sites()))
    change(arrays)
    with pytest.raises(InvalidInputError, match=message):
        bundle_from_arrays(arrays)
