import json

import numpy as np
import pytest

from evaku_ai.adapters import bundle
from evaku_ai.core.errors import DataNotLoadedError, InvalidInputError
from evaku_ai.engine.bundle import bundle_to_arrays
from evaku_ai.engine.synthetic import synthetic_geometry, synthetic_scenario
from tests.fixtures.evacuation import sample_sites

VERSION = "2026-10-16.1"


def _write(tmp_path, version=VERSION):
    return bundle.write_bundle(
        synthetic_scenario(),
        sample_sites(),
        synthetic_geometry(synthetic_scenario()),
        tmp_path / "static",
        data_version=version,
        region_id="aceh-tamiang",
        sources={"hand.tif": "a" * 64},
    )


def _edit_manifest(path, **changes):
    manifest = json.loads((path / bundle.MANIFEST_FILE).read_text())
    manifest.update(changes)
    (path / bundle.MANIFEST_FILE).write_text(json.dumps(manifest))


def test_round_trip(tmp_path):
    path = _write(tmp_path)
    assert path == tmp_path / "static" / VERSION
    assert sorted(p.name for p in (tmp_path / "static").iterdir()) == [VERSION]
    loaded = bundle.read_bundle(path, VERSION)
    assert loaded.manifest.region_id == "aceh-tamiang"
    assert loaded.manifest.counts["edges"] == synthetic_scenario().graph.n_edges
    assert loaded.manifest.counts["evacuation_sites"] == 2
    original = bundle_to_arrays(
        synthetic_scenario(), sample_sites(), synthetic_geometry(synthetic_scenario())
    )
    for key, values in bundle_to_arrays(loaded.data, loaded.sites, loaded.geometry).items():
        np.testing.assert_array_equal(values, original[key])


def test_existing_version_is_never_overwritten(tmp_path):
    _write(tmp_path)
    with pytest.raises(InvalidInputError, match="never overwritten"):
        _write(tmp_path)


@pytest.mark.parametrize("version", ["../etc", "a/b", "", ".."])
def test_invalid_versions_are_rejected(tmp_path, version):
    with pytest.raises(InvalidInputError, match="invalid data version"):
        _write(tmp_path, version)
    with pytest.raises(DataNotLoadedError, match="invalid data version"):
        bundle.read_bundle(tmp_path, version)


def test_missing_bundle(tmp_path):
    with pytest.raises(DataNotLoadedError, match="no static data bundle"):
        bundle.read_bundle(tmp_path / "static" / VERSION, VERSION)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"data_version": "2026-10-01.1"}, "expected 2026-10-16.1"),
        ({"format": 99}, "format 99"),
        ({"format": 1}, "rebuild the bundle"),
        ({"surprise": 1}, "invalid"),
        ({"sha256": "0" * 64}, "checksum"),
    ],
)
def test_manifest_problems(tmp_path, changes, message):
    path = _write(tmp_path)
    _edit_manifest(path, **changes)
    with pytest.raises(DataNotLoadedError, match=message):
        bundle.read_bundle(path, VERSION)


def test_broken_manifest_and_missing_arrays(tmp_path):
    path = _write(tmp_path)
    (path / bundle.ARRAYS_FILE).unlink()
    with pytest.raises(DataNotLoadedError, match="missing"):
        bundle.read_bundle(path, VERSION)
    (path / bundle.MANIFEST_FILE).write_text("{not json")
    with pytest.raises(DataNotLoadedError, match="invalid"):
        bundle.read_bundle(path, VERSION)


def test_size_limits(tmp_path, monkeypatch):
    path = _write(tmp_path)
    with pytest.raises(DataNotLoadedError, match="larger than"):
        bundle.read_bundle(path, VERSION, max_bytes=10)
    monkeypatch.setattr(bundle, "MAX_MANIFEST_BYTES", 10)
    with pytest.raises(DataNotLoadedError, match="larger than 10"):
        bundle.read_bundle(path, VERSION)


def _replace_arrays(path, **arrays):
    target = path / bundle.ARRAYS_FILE
    np.savez_compressed(target, **arrays)
    _edit_manifest(path, sha256=bundle.file_sha256(target))


def test_pickled_objects_are_refused(tmp_path):
    path = _write(tmp_path)
    arrays = dict(
        bundle_to_arrays(
            synthetic_scenario(), sample_sites(), synthetic_geometry(synthetic_scenario())
        )
    )
    arrays["rivers"] = np.array([{"river": 1}], dtype=object)
    _replace_arrays(path, **arrays)
    with pytest.raises(DataNotLoadedError, match="invalid"):
        bundle.read_bundle(path, VERSION)


def test_arrays_that_do_not_form_a_scenario_are_refused(tmp_path):
    path = _write(tmp_path)
    arrays = dict(
        bundle_to_arrays(
            synthetic_scenario(), sample_sites(), synthetic_geometry(synthetic_scenario())
        )
    )
    arrays.pop("hand_min")
    _replace_arrays(path, **arrays)
    with pytest.raises(DataNotLoadedError, match="hand_min"):
        bundle.read_bundle(path, VERSION)
