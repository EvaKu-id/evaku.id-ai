import hashlib
import json
from datetime import UTC, datetime

import pytest

from evaku_ai.adapters import release
from evaku_ai.adapters.release import (
    LATEST_KEY,
    MAX_FILE_BYTES,
    MAX_RELEASE_JSON_BYTES,
    ReleaseFile,
    ReleaseManifest,
    check_release_version,
    content_type_for,
    describe_files,
    fetch_bundle,
    parse_latest,
    parse_release,
    read_latest,
    release_prefix,
    verify_release,
    version_key,
)
from evaku_ai.core.errors import DataNotLoadedError, InvalidInputError
from tests.fixtures.storage import FakeStorage

VERSION = "2026-10-16.1"
FILES = {
    "manifest.json": b'{"format": 2}',
    "scenario.npz": b"npz-bytes",
    "README.md": b"# readme",
    "geo/villages.parquet": b"parquet-bytes",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _entry(path: str, data: bytes) -> ReleaseFile:
    return ReleaseFile(
        path=path,
        size=len(data),
        sha256=_sha(data),
        content_type=release.content_type_for(release.Path(path)),
    )


def _manifest(files=FILES, version=VERSION, **changes) -> ReleaseManifest:
    values = {
        "format": 1,
        "data_version": version,
        "region_id": "aceh-tamiang",
        "created_at": datetime(2026, 10, 16, tzinfo=UTC),
        "git_commit": "a" * 40,
        "git_dirty": False,
        "bundle_format": 2,
        "licenses": ["CC BY 4.0"],
        "files": [_entry(path, data) for path, data in files.items()],
    }
    return ReleaseManifest(**(values | changes))


def _remote(files=FILES, version=VERSION, manifest=None) -> FakeStorage:
    storage = FakeStorage()
    prefix = release_prefix(version)
    for path, data in files.items():
        storage.objects[prefix + path] = (data, "application/octet-stream", _sha(data))
    body = (manifest or _manifest(files, version)).model_dump_json().encode()
    storage.objects[prefix + "release.json"] = (body, "application/json", _sha(body))
    return storage


def test_release_versions():
    assert check_release_version("2026-10-16.2") == "2026-10-16.2"
    for bad in ("2026-10-16", "../x", "v1", "2026-13-45.1", "2026-10-16.2/../x", ""):
        with pytest.raises(InvalidInputError):
            check_release_version(bad)


def test_versions_compare_by_date_and_number():
    assert version_key("2026-10-16.10") > version_key("2026-10-16.9") > version_key("2026-10-15.99")


def test_prefix():
    assert release_prefix(VERSION) == "data/static/2026-10-16.1/"


def test_content_types():
    expected = {
        "a.parquet": "application/vnd.apache.parquet",
        "a.tif": "image/tiff",
        "a.json": "application/json",
        "a.csv": "text/csv",
        "a.md": "text/markdown",
        "a.npz": "application/octet-stream",
    }
    for name, kind in expected.items():
        assert content_type_for(release.Path(name)) == kind
    with pytest.raises(InvalidInputError, match="unexpected file type"):
        content_type_for(release.Path("a.exe"))


def test_describe_files_is_recursive_sorted_and_hashed(tmp_path):
    (tmp_path / "geo").mkdir()
    (tmp_path / "b.json").write_bytes(b"{}")
    (tmp_path / "geo" / "a.csv").write_bytes(b"x,y")
    (tmp_path / "release.json").write_bytes(b"skip")
    found = describe_files(tmp_path, exclude={"release.json"})
    assert [f.path for f in found] == ["b.json", "geo/a.csv"]
    assert found[1].sha256 == _sha(b"x,y") and found[1].size == 3
    assert found[1].content_type == "text/csv"


def test_describe_files_rejects_unexpected_types(tmp_path):
    (tmp_path / "x.exe").write_bytes(b"MZ")
    with pytest.raises(InvalidInputError):
        describe_files(tmp_path, exclude=set())


def _payload(**changes) -> bytes:
    data = json.loads(_manifest().model_dump_json())
    return json.dumps(data | changes).encode()


def test_parse_release_round_trips():
    assert parse_release(_payload()) == _manifest()


@pytest.mark.parametrize(
    "changes",
    [
        {"surprise": 1},
        {"format": 2},
        {"data_version": "v1"},
        {
            "files": [
                {
                    "path": "../x.json",
                    "size": 1,
                    "sha256": "a" * 64,
                    "content_type": "application/json",
                }
            ]
        },
        {
            "files": [
                {
                    "path": "/x.json",
                    "size": 1,
                    "sha256": "a" * 64,
                    "content_type": "application/json",
                }
            ]
        },
        {
            "files": [
                {
                    "path": "a/./b.json",
                    "size": 1,
                    "sha256": "a" * 64,
                    "content_type": "application/json",
                }
            ]
        },
        {
            "files": [
                {"path": "x.json", "size": 1, "sha256": "XYZ", "content_type": "application/json"}
            ]
        },
        {"files": [{"path": "x.json", "size": 1, "sha256": "a" * 64, "content_type": "text/evil"}]},
        {
            "files": [
                {
                    "path": "release.json",
                    "size": 1,
                    "sha256": "a" * 64,
                    "content_type": "application/json",
                }
            ]
        },
        {
            "files": [
                {
                    "path": "x.json",
                    "size": 1,
                    "sha256": "a" * 64,
                    "content_type": "application/json",
                },
                {
                    "path": "x.json",
                    "size": 1,
                    "sha256": "a" * 64,
                    "content_type": "application/json",
                },
            ]
        },
    ],
)
def test_parse_release_rejects_bad_manifests(changes):
    with pytest.raises(InvalidInputError):
        parse_release(_payload(**changes))


def test_parse_release_rejects_oversized_and_broken_input():
    with pytest.raises(InvalidInputError, match="larger than"):
        parse_release(b" " * (MAX_RELEASE_JSON_BYTES + 1))
    with pytest.raises(InvalidInputError, match="invalid"):
        parse_release(b"{not json")


def test_latest_pointer(tmp_path):
    pointer = json.dumps(
        {
            "data_version": VERSION,
            "released_at": "2026-10-16T10:00:00Z",
            "prefix": release_prefix(VERSION),
        }
    ).encode()
    assert parse_latest(pointer).data_version == VERSION
    with pytest.raises(InvalidInputError, match=r"latest\.json"):
        parse_latest(b'{"data_version": "v1"}')
    storage = FakeStorage()
    assert read_latest(storage, tmp_path) is None
    storage.objects[LATEST_KEY] = (pointer, "application/json", _sha(pointer))
    assert read_latest(storage, tmp_path).prefix == release_prefix(VERSION)


def test_fetch_bundle_downloads_only_the_api_files(tmp_path):
    storage = _remote()
    dest = tmp_path / "static" / VERSION
    assert fetch_bundle(storage, VERSION, dest) == dest
    assert sorted(p.name for p in dest.iterdir()) == [
        "manifest.json",
        "release.json",
        "scenario.npz",
    ]
    assert (dest / "scenario.npz").read_bytes() == FILES["scenario.npz"]
    downloaded = [key for call, key in storage.calls if call == "download"]
    assert not any("geo/" in key or key.endswith("README.md") for key in downloaded)
    assert sorted(p.name for p in dest.parent.iterdir()) == [VERSION]


def test_fetch_bundle_replaces_an_empty_destination(tmp_path):
    dest = tmp_path / VERSION
    dest.mkdir()
    fetch_bundle(_remote(), VERSION, dest)
    assert (dest / "manifest.json").is_file()


def test_fetch_bundle_never_overwrites_existing_data(tmp_path):
    dest = tmp_path / VERSION
    dest.mkdir()
    (dest / "keep.txt").write_text("mine")
    with pytest.raises(DataNotLoadedError, match="remove it first"):
        fetch_bundle(_remote(), VERSION, dest)
    assert (dest / "keep.txt").read_text() == "mine"


def test_fetch_bundle_rejects_a_bad_version(tmp_path):
    with pytest.raises(DataNotLoadedError, match="release version"):
        fetch_bundle(_remote(), "dev", tmp_path / "dev")


def _corrupt(storage, path, data):
    key = release_prefix(VERSION) + path
    _, kind, sha = storage.objects[key]
    storage.objects[key] = (data, kind, sha)


def test_fetch_bundle_failures_leave_nothing_behind(tmp_path):
    cases = {}
    cases["no release.json"] = FakeStorage()
    wrong = _remote()
    wrong.objects[release_prefix(VERSION) + "release.json"] = _remote(
        version="2026-10-15.1"
    ).objects[release_prefix("2026-10-15.1") + "release.json"]
    cases["other version"] = wrong
    bad_sha = _remote()
    _corrupt(bad_sha, "scenario.npz", b"tampered!!")
    cases["checksum"] = bad_sha
    short = _remote()
    _corrupt(short, "scenario.npz", b"short")
    cases["size"] = short
    missing_entry = _remote(
        manifest=_manifest({k: v for k, v in FILES.items() if k != "scenario.npz"})
    )
    cases["entry missing"] = missing_entry
    too_big = _remote(
        manifest=_manifest(
            files={"manifest.json": FILES["manifest.json"]},
        ).model_copy(
            update={
                "files": [
                    _entry("manifest.json", FILES["manifest.json"]),
                    ReleaseFile(
                        path="scenario.npz",
                        size=MAX_FILE_BYTES + 1,
                        sha256="a" * 64,
                        content_type="application/octet-stream",
                    ),
                ]
            }
        )
    )
    cases["too big"] = too_big
    for name, storage in cases.items():
        dest = tmp_path / name.replace(" ", "-") / VERSION
        with pytest.raises(DataNotLoadedError):
            fetch_bundle(storage, VERSION, dest)
        assert not dest.exists(), name
        assert list(dest.parent.iterdir()) == [], name


def test_verify_release_downloads_and_checks_everything(tmp_path):
    manifest = verify_release(_remote(), VERSION, tmp_path / "verify")
    assert [f.path for f in manifest.files] == list(FILES)
    for path, data in FILES.items():
        assert (tmp_path / "verify" / path).read_bytes() == data


def test_verify_release_names_the_corrupt_file(tmp_path):
    storage = _remote()
    _corrupt(storage, "geo/villages.parquet", b"parquet-bytez")
    with pytest.raises(InvalidInputError, match=r"geo/villages\.parquet"):
        verify_release(storage, VERSION, tmp_path / "verify")


def test_verify_release_reports_a_missing_object(tmp_path):
    storage = _remote()
    del storage.objects[release_prefix(VERSION) + "README.md"]
    with pytest.raises(InvalidInputError, match=r"README\.md"):
        verify_release(storage, VERSION, tmp_path / "verify")


def test_verify_release_flags_unexpected_objects(tmp_path):
    storage = _remote()
    storage.objects[release_prefix(VERSION) + "geo/surprise.parquet"] = (
        b"x",
        "text/csv",
        _sha(b"x"),
    )
    with pytest.raises(InvalidInputError, match="unexpected objects"):
        verify_release(storage, VERSION, tmp_path / "verify")


def test_verify_release_checks_the_version(tmp_path):
    storage = _remote()
    storage.objects[release_prefix(VERSION) + "release.json"] = _remote(
        version="2026-10-15.1"
    ).objects[release_prefix("2026-10-15.1") + "release.json"]
    with pytest.raises(InvalidInputError, match="is version"):
        verify_release(storage, VERSION, tmp_path / "verify")
