import subprocess

import pytest

from evaku_ai.adapters import tippecanoe
from evaku_ai.core.errors import ExternalToolError


def _done(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def test_find_binary_prefers_the_pinned_build(tmp_path, monkeypatch):
    binary = tmp_path / "tippecanoe"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    monkeypatch.setattr(tippecanoe.shutil, "which", lambda name: "/usr/bin/tippecanoe")
    assert tippecanoe.find_binary(tools_dir=tmp_path) == binary


def test_find_binary_falls_back_to_path(tmp_path, monkeypatch):
    (tmp_path / "tippecanoe").write_text("not executable")
    monkeypatch.setattr(tippecanoe.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert str(tippecanoe.find_binary(tools_dir=tmp_path)) == "/usr/bin/tippecanoe"


def test_find_binary_explains_how_to_install(tmp_path, monkeypatch):
    monkeypatch.setattr(tippecanoe.shutil, "which", lambda name: None)
    with pytest.raises(ExternalToolError, match="make tippecanoe"):
        tippecanoe.find_binary(tools_dir=tmp_path)


def test_version_reads_stderr(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return _done(stderr="tippecanoe v2.79.0\n")

    monkeypatch.setattr(tippecanoe.subprocess, "run", fake_run)
    assert tippecanoe.version(tmp_path / "tc") == "tippecanoe v2.79.0"
    args, kwargs = calls[0]
    assert args == [str(tmp_path / "tc"), "--version"]
    assert kwargs["timeout"] == tippecanoe.VERSION_TIMEOUT_S


def test_version_without_output_is_unknown(monkeypatch, tmp_path):
    monkeypatch.setattr(tippecanoe.subprocess, "run", lambda args, **kw: _done())
    assert tippecanoe.version(tmp_path / "tc") == "unknown"


@pytest.mark.parametrize(
    "outcome",
    [_done(returncode=1), subprocess.TimeoutExpired(["tc"], 60)],
)
def test_version_failures_raise(monkeypatch, tmp_path, outcome):
    def fake_run(args, **kwargs):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(tippecanoe.subprocess, "run", fake_run)
    with pytest.raises(ExternalToolError, match="--version"):
        tippecanoe.version(tmp_path / "tc")


def test_run_returns_the_output(monkeypatch, tmp_path):
    output = tmp_path / "out.pmtiles"

    def fake_run(args, **kwargs):
        assert kwargs["timeout"] == tippecanoe.RUN_TIMEOUT_S
        output.write_bytes(b"tiles")
        return _done()

    monkeypatch.setattr(tippecanoe.subprocess, "run", fake_run)
    assert tippecanoe.run(["tc", "in.geojsonl"], output) == output


def test_run_reports_stderr_on_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        tippecanoe.subprocess, "run", lambda args, **kw: _done(returncode=2, stderr="bad input")
    )
    with pytest.raises(ExternalToolError, match="failed \\(2\\): bad input"):
        tippecanoe.run(["tc"], tmp_path / "out.pmtiles")


def test_run_without_output_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(tippecanoe.subprocess, "run", lambda args, **kw: _done(stdout="ok"))
    with pytest.raises(ExternalToolError, match="failed \\(0\\): ok"):
        tippecanoe.run(["tc"], tmp_path / "out.pmtiles")


def test_run_timeout_fails(monkeypatch, tmp_path):
    def fake_run(args, **kwargs):
        raise subprocess.TimeoutExpired(args, 1)

    monkeypatch.setattr(tippecanoe.subprocess, "run", fake_run)
    with pytest.raises(ExternalToolError, match="timed out"):
        tippecanoe.run(["tc"], tmp_path / "out.pmtiles")
