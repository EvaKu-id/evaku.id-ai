import io
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest
import zstandard

from evaku_ai.adapters import whitebox
from evaku_ai.core.errors import ExternalToolError


def _conda(
    path: Path, files: dict[str, bytes], *, payloads: int = 1, symlink: bool = False
) -> Path:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            if symlink:
                info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
            else:
                info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    compressed = zstandard.ZstdCompressor().compress(raw.getvalue())
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("metadata.json", b"{}")
        for i in range(payloads):
            archive.writestr(f"pkg-wbt-{i}.tar.zst", compressed)
    return path


def test_install_extracts_only_the_binary(tmp_path):
    package = _conda(
        tmp_path / "wbt.conda",
        {"bin/whitebox_gui": b"gui", "bin/whitebox_tools": b"#!binary", "lib/x.py": b"y"},
    )
    binary = whitebox.install(package, tmp_path / "bin")
    assert binary.read_bytes() == b"#!binary"
    assert binary.stat().st_mode & 0o100
    assert sorted(p.name for p in (tmp_path / "bin").iterdir()) == ["whitebox_tools"]
    assert whitebox.install(tmp_path / "missing.conda", tmp_path / "bin") == binary


@pytest.mark.parametrize(
    ("files", "options", "message"),
    [
        ({"bin/other": b"x"}, {}, "not found"),
        ({"../bin/whitebox_tools": b"x"}, {}, "not found"),
        ({"bin/whitebox_tools": b""}, {"symlink": True}, "not found"),
        ({"bin/whitebox_tools": b"x"}, {"payloads": 2}, "expected one"),
        ({"bin/whitebox_tools": b"x"}, {"payloads": 0}, "expected one"),
    ],
)
def test_install_rejects_bad_packages(tmp_path, files, options, message):
    package = _conda(tmp_path / "wbt.conda", files, **options)
    with pytest.raises(ExternalToolError, match=message):
        whitebox.install(package, tmp_path / "bin")
    assert not (tmp_path / "bin" / "whitebox_tools").exists()


def test_command_is_an_argument_list():
    cmd = whitebox.command(
        Path("/opt/wbt"),
        "ExtractStreams",
        {"flow_accum": Path("a b.tif"), "threshold": 5, "zero_background": True, "log": False},
    )
    assert cmd == [
        "/opt/wbt",
        "--run=ExtractStreams",
        "--flow_accum=a b.tif",
        "--threshold=5",
        "--zero_background",
    ]


class _Recorder:
    def __init__(self, returncode=0, create=None, raises=None):
        self.calls = []
        self.returncode, self.create, self.raises = returncode, create, raises

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        if self.raises:
            raise self.raises
        if self.create:
            self.create.write_bytes(b"tif")
        return subprocess.CompletedProcess(
            cmd, self.returncode, stdout="WhiteboxTools v2.4.0\n", stderr="boom"
        )


def test_run_without_shell_and_returns_output(tmp_path, monkeypatch):
    out = tmp_path / "streams.tif"
    recorder = _Recorder(create=out)
    monkeypatch.setattr(whitebox.subprocess, "run", recorder)
    result = whitebox.extract_streams(Path("/opt/wbt"), tmp_path / "acc.tif", out, 1112)
    assert result == out
    cmd, kwargs = recorder.calls[0]
    assert isinstance(cmd, list)
    assert "shell" not in kwargs
    assert kwargs["timeout"] == whitebox.RUN_TIMEOUT_S
    assert "--threshold=1112" in cmd


@pytest.mark.parametrize(
    ("recorder", "message"),
    [
        (_Recorder(returncode=1), "failed \\(1\\): boom"),
        (_Recorder(returncode=0), "failed \\(0\\)"),
        (_Recorder(raises=subprocess.TimeoutExpired("wbt", 1)), "timed out"),
    ],
)
def test_run_errors(tmp_path, monkeypatch, recorder, message):
    monkeypatch.setattr(whitebox.subprocess, "run", recorder)
    with pytest.raises(ExternalToolError, match=message):
        whitebox.d8_pointer(Path("/opt/wbt"), tmp_path / "dem.tif", tmp_path / "d8.tif")


def test_tool_wrappers_pass_expected_flags(tmp_path, monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(whitebox.subprocess, "run", recorder)
    wbt, a, b = Path("/opt/wbt"), tmp_path / "a.tif", tmp_path / "b.tif"
    calls = [
        (
            lambda o: whitebox.breach(wbt, a, o, 100),
            ["--run=BreachDepressionsLeastCost", "--dist=100", "--fill"],
        ),
        (lambda o: whitebox.flow_accumulation(wbt, a, o), ["--out_type=cells", "--pntr"]),
        (lambda o: whitebox.elevation_above_stream(wbt, a, b, o), ["--run=ElevationAboveStream"]),
        (lambda o: whitebox.watershed(wbt, a, b, o), [f"--pour_pts={b}"]),
        (lambda o: whitebox.streams_to_vector(wbt, a, b, o), ["--run=RasterStreamsToVector"]),
    ]
    for i, (call, expected) in enumerate(calls):
        out = tmp_path / f"out{i}.tif"
        recorder.create = out
        assert call(out) == out
        assert set(expected) <= set(recorder.calls[-1][0])


def test_version(monkeypatch):
    monkeypatch.setattr(whitebox.subprocess, "run", _Recorder())
    assert whitebox.version(Path("/opt/wbt")) == "WhiteboxTools v2.4.0"
    monkeypatch.setattr(whitebox.subprocess, "run", _Recorder(returncode=2))
    with pytest.raises(ExternalToolError, match="version"):
        whitebox.version(Path("/opt/wbt"))
