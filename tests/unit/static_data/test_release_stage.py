import copy
import hashlib
import json
import logging
import subprocess

import pandas as pd
import pytest
import yaml

from evaku_ai.adapters.release import (
    LATEST_KEY,
    RELEASE_FILE,
    describe_files,
    parse_latest,
    parse_release,
    release_prefix,
)
from evaku_ai.core.config import Settings
from evaku_ai.core.errors import ConfigurationError, InvalidInputError, StorageError
from evaku_ai.static_data import release as module
from evaku_ai.static_data.release import (
    FEATURES_FILE,
    GEO_DIR,
    GEO_FILES,
    LICENSES,
    META_FILES,
    git_state,
    stage,
    upload,
)
from tests.fixtures.processed import make_release_bundle, write_processed
from tests.fixtures.region import VALID
from tests.fixtures.storage import FakeStorage

VERSION = "2026-10-16.2"
COMMIT = "c" * 40


@pytest.fixture(autouse=True)
def _clean_git(monkeypatch):
    monkeypatch.setattr(module, "git_state", lambda: (COMMIT, False))


@pytest.fixture
def workspace(tmp_path):
    processed = write_processed(tmp_path / "processed")
    static = tmp_path / "static"
    make_release_bundle(processed, static, VERSION)
    region = tmp_path / "aceh-tamiang.yaml"
    region.write_text(yaml.safe_dump(copy.deepcopy(VALID), sort_keys=False), encoding="utf-8")
    return tmp_path, processed, static, region


def _stage(workspace, version=VERSION):
    _, processed, static, region = workspace
    return stage(version, region, processed, static)


def test_stage_builds_the_full_release_folder(workspace):
    _, _, static, _ = workspace
    manifest = _stage(workspace)
    root = static / VERSION
    assert sorted(p.name for p in (root / GEO_DIR).iterdir()) == sorted(
        [*GEO_FILES, *META_FILES, FEATURES_FILE]
    )
    assert (root / "release.json").is_file() and not list(root.glob("*.part"))
    assert parse_release((root / "release.json").read_bytes()) == manifest
    assert manifest.files == describe_files(root, exclude={RELEASE_FILE})
    assert {f.path for f in manifest.files} >= {"manifest.json", "scenario.npz", "README.md"}
    assert (manifest.git_commit, manifest.git_dirty) == (COMMIT, False)
    assert manifest.bundle_format == 3 and manifest.region_id == "aceh-tamiang"
    assert manifest.licenses == list(LICENSES)
    features = pd.read_csv(root / GEO_DIR / FEATURES_FILE, dtype={"village_id": str})
    assert features["village_id"].tolist() == ["1114000001", "1114000002"]


def test_stage_writes_a_readme_with_version_files_and_credits(workspace):
    _, _, static, _ = workspace
    manifest = _stage(workspace)
    readme = (static / VERSION / "README.md").read_text(encoding="utf-8")
    assert VERSION in readme and LATEST_KEY in readme
    assert all(credit in readme for credit in LICENSES)
    assert "`geo/hand.tif`" in readme and "`village_id`" in readme
    assert "README.md" in {f.path for f in manifest.files}


def test_stage_records_a_dirty_tree_and_warns(workspace, monkeypatch, caplog):
    monkeypatch.setattr(module, "git_state", lambda: (None, True))
    with caplog.at_level(logging.WARNING):
        manifest = _stage(workspace)
    assert (manifest.git_commit, manifest.git_dirty) == (None, True)
    assert "uncommitted changes" in caplog.text


def test_stage_replaces_leftovers_of_a_failed_staging(workspace):
    _, _, static, _ = workspace
    (static / VERSION / GEO_DIR).mkdir()
    (static / VERSION / GEO_DIR / "junk.parquet").write_bytes(b"x")
    _stage(workspace)
    assert not (static / VERSION / GEO_DIR / "junk.parquet").exists()


def test_stage_needs_the_bundle_first(workspace):
    _, processed, static, region = workspace
    with pytest.raises(InvalidInputError, match="build the bundle first"):
        stage("2026-10-16.3", region, processed, static)


def test_stage_rejects_processed_data_that_changed_after_the_bundle(workspace):
    _, processed, _, _ = workspace
    pd.DataFrame({"id": [9]}).to_parquet(processed / "edge_hand.parquet")
    with pytest.raises(InvalidInputError, match=r"edge_hand\.parquet.*changed since bundle"):
        _stage(workspace)


def test_stage_rejects_an_already_staged_release(workspace):
    _stage(workspace)
    with pytest.raises(InvalidInputError, match="already staged"):
        _stage(workspace)


def test_stage_lists_missing_processed_files(workspace):
    _, processed, _, _ = workspace
    (processed / "hand.meta.json").unlink()
    with pytest.raises(InvalidInputError, match=r"hand\.meta\.json"):
        _stage(workspace)


def test_stage_rejects_a_bad_version(workspace):
    with pytest.raises(InvalidInputError, match="release version"):
        _stage(workspace, "v1")


def _staged(workspace):
    _, _, static, _ = workspace
    _stage(workspace)
    return static


def test_upload_sends_data_first_then_the_release_file_then_latest(workspace):
    static = _staged(workspace)
    storage = FakeStorage()
    upload(VERSION, storage, static)
    prefix = release_prefix(VERSION)
    sent = [key for call, key in storage.calls if call == "upload"]
    assert sent[-2:] == [prefix + RELEASE_FILE, LATEST_KEY]
    assert all(key.startswith(prefix) and key != prefix + RELEASE_FILE for key in sent[:-2])
    manifest = parse_release((static / VERSION / RELEASE_FILE).read_bytes())
    assert len(sent) == len(manifest.files) + 2
    for entry in manifest.files:
        data, kind, sha = storage.objects[prefix + entry.path]
        assert (kind, sha, len(data)) == (entry.content_type, entry.sha256, entry.size)
    assert parse_latest(storage.objects[LATEST_KEY][0]).data_version == VERSION


def test_upload_refuses_to_overwrite_a_release(workspace):
    static = _staged(workspace)
    storage = FakeStorage()
    upload(VERSION, storage, static)
    with pytest.raises(InvalidInputError, match="never overwritten"):
        upload(VERSION, storage, static, resume=True)


def test_upload_needs_a_staged_release(workspace):
    _, _, static, _ = workspace
    with pytest.raises(InvalidInputError, match="not staged"):
        upload(VERSION, FakeStorage(), static)


def test_upload_detects_files_changed_after_staging(workspace):
    static = _staged(workspace)
    (static / VERSION / GEO_DIR / "streams.parquet").write_bytes(b"changed")
    with pytest.raises(InvalidInputError, match="changed since staging"):
        upload(VERSION, FakeStorage(), static)


def test_a_failed_upload_can_be_resumed_without_resending_files(workspace):
    static = _staged(workspace)
    prefix = release_prefix(VERSION)
    failing = prefix + "geo/streams.parquet"
    storage = FakeStorage(fail_on=(failing,))
    with pytest.raises(StorageError):
        upload(VERSION, storage, static)
    assert prefix + RELEASE_FILE not in storage.objects and LATEST_KEY not in storage.objects
    with pytest.raises(InvalidInputError, match="--resume"):
        upload(VERSION, storage, static)

    already = {key for key in storage.objects}
    storage.fail_on.clear()
    storage.calls.clear()
    upload(VERSION, storage, static, resume=True)
    resent = {key for call, key in storage.calls if call == "upload"}
    assert not resent & already
    assert failing in resent and prefix + RELEASE_FILE in resent


def test_resume_refuses_a_remote_file_that_differs(workspace):
    static = _staged(workspace)
    prefix = release_prefix(VERSION)
    storage = FakeStorage()
    storage.objects[prefix + "README.md"] = (b"other", "text/markdown", "0" * 64)
    with pytest.raises(InvalidInputError, match="differs from the staged file"):
        upload(VERSION, storage, static, resume=True)
    assert storage.objects[prefix + "README.md"][0] == b"other"


def _latest(storage, version):
    payload = json.dumps(
        {
            "data_version": version,
            "released_at": "2026-10-01T00:00:00Z",
            "prefix": release_prefix(version),
        }
    ).encode()
    storage.objects[LATEST_KEY] = (payload, "application/json", hashlib.sha256(payload).hexdigest())


def test_latest_moves_forward_only(workspace):
    static = _staged(workspace)
    older, newer = FakeStorage(), FakeStorage()
    _latest(older, "2026-10-15.9")
    _latest(newer, "2026-10-17.1")
    upload(VERSION, older, static)
    upload(VERSION, newer, static)
    assert parse_latest(older.objects[LATEST_KEY][0]).data_version == VERSION
    assert parse_latest(newer.objects[LATEST_KEY][0]).data_version == "2026-10-17.1"


def _run_env(workspace, monkeypatch, storage):
    tmp, processed, static, region = workspace
    monkeypatch.chdir(tmp)
    (tmp / "data").mkdir()
    (tmp / "data" / "processed").symlink_to(processed)
    (tmp / "data" / "static").symlink_to(static)
    settings = Settings(_env_file=None, r2_bucket_private="evaku-private")
    monkeypatch.setattr(module, "get_settings", lambda: settings)
    monkeypatch.setattr(module, "r2_storage", lambda s, bucket: storage)
    return region


def test_run_stages_then_uploads_and_verify_checks_it(workspace, monkeypatch, tmp_path):
    storage = FakeStorage()
    region = _run_env(workspace, monkeypatch, storage)
    module.run(VERSION, region)
    assert release_prefix(VERSION) + RELEASE_FILE in storage.objects
    manifest = module.verify(VERSION, dest_root=tmp_path / "verify")
    assert manifest.data_version == VERSION
    assert (tmp_path / "verify" / VERSION / GEO_DIR / "hand.tif").is_file()


def test_run_again_is_refused(workspace, monkeypatch):
    storage = FakeStorage()
    region = _run_env(workspace, monkeypatch, storage)
    module.run(VERSION, region)
    with pytest.raises(InvalidInputError, match="never overwritten"):
        module.run(VERSION, region)


def test_run_without_r2_settings_fails_before_staging(workspace, monkeypatch):
    tmp, _, static, region = workspace
    monkeypatch.chdir(tmp)
    monkeypatch.setattr(module, "get_settings", lambda: Settings(_env_file=None))
    with pytest.raises(ConfigurationError, match="must be set"):
        module.run(VERSION, region)
    assert not (static / VERSION / RELEASE_FILE).exists()


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)  # noqa: S603, S607


def test_git_state_reads_commit_and_cleanliness(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.txt").write_text("a")
    _git(tmp_path, "add", "a.txt")
    _git(
        tmp_path,
        "-c", "user.email=t@example.org", "-c", "user.name=t", "-c", "commit.gpgsign=false",
        "commit", "-q", "-m", "x",
    )  # fmt: skip
    commit, dirty = git_state(tmp_path)
    assert commit is not None and len(commit) == 40 and dirty is False
    (tmp_path / "a.txt").write_text("changed")
    assert git_state(tmp_path) == (commit, True)


def test_git_state_without_a_repository_or_git(tmp_path, monkeypatch, caplog):
    with caplog.at_level(logging.WARNING):
        assert git_state(tmp_path) == (None, True)
    assert "not a git checkout" in caplog.text

    def missing(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(module.subprocess, "run", missing)
    assert git_state(tmp_path) == (None, True)
