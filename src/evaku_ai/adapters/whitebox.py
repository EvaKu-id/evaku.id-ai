import logging
import stat
import subprocess
import tarfile
import zipfile
from pathlib import Path

import zstandard

from evaku_ai.core.errors import ExternalToolError
from evaku_ai.domain.region import RemoteFile

log = logging.getLogger(__name__)

BINARY_MEMBER = "bin/whitebox_tools"
RUN_TIMEOUT_S = 3600.0
STDERR_TAIL = 2000
WHITEBOX_TOOLS = RemoteFile(
    path=Path("data/tools/whitebox_tools-2.4.0-py311ha6e1976_4.conda"),
    url="https://conda.anaconda.org/conda-forge/linux-64/whitebox_tools-2.4.0-py311ha6e1976_4.conda",
    sha256="1f536690a5bef4866cdca007b240843f2bcac12c4a6745def43dced58724467e",
    source="WhiteboxTools v2.4.0 (conda-forge build)",
    license="MIT",
)

ToolArg = str | int | float | bool | Path


def install(package: Path, dest_dir: Path) -> Path:
    binary = dest_dir / "whitebox_tools"
    if binary.is_file():
        return binary
    data = _read_binary(package)
    dest_dir.mkdir(parents=True, exist_ok=True)
    part = binary.with_name(binary.name + ".part")
    part.write_bytes(data)
    part.chmod(part.stat().st_mode | stat.S_IXUSR)
    part.replace(binary)
    return binary


def command(binary: Path, tool: str, args: dict[str, ToolArg]) -> list[str]:
    cmd = [str(binary), f"--run={tool}"]
    for key, value in args.items():
        if value is True:
            cmd.append(f"--{key}")
        elif value is not False:
            cmd.append(f"--{key}={value}")
    return cmd


def run(binary: Path, tool: str, args: dict[str, ToolArg], output: Path) -> Path:
    cmd = command(binary, tool, args)
    log.info("whitebox %s", tool)
    try:
        # Argument list without a shell; the binary comes from a checksum-pinned archive.
        done = subprocess.run(  # noqa: S603
            cmd, capture_output=True, text=True, timeout=RUN_TIMEOUT_S, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise ExternalToolError(f"whitebox {tool} timed out after {RUN_TIMEOUT_S:.0f} s") from exc
    if done.returncode != 0 or not output.is_file():
        tail = (done.stderr or done.stdout)[-STDERR_TAIL:]
        raise ExternalToolError(f"whitebox {tool} failed ({done.returncode}): {tail}")
    return output


def version(binary: Path) -> str:
    done = subprocess.run(  # noqa: S603
        [str(binary), "--version"], capture_output=True, text=True, timeout=60.0, check=False
    )
    if done.returncode != 0:
        raise ExternalToolError(f"whitebox --version failed ({done.returncode})")
    return done.stdout.strip().splitlines()[0]


def breach(binary: Path, dem: Path, output: Path, max_dist_cells: int) -> Path:
    args: dict[str, ToolArg] = {"dem": dem, "output": output, "dist": max_dist_cells, "fill": True}
    return run(binary, "BreachDepressionsLeastCost", args, output)


def d8_pointer(binary: Path, dem: Path, output: Path) -> Path:
    return run(binary, "D8Pointer", {"dem": dem, "output": output}, output)


def flow_accumulation(binary: Path, pointer: Path, output: Path) -> Path:
    args: dict[str, ToolArg] = {
        "input": pointer,
        "output": output,
        "out_type": "cells",
        "pntr": True,
    }
    return run(binary, "D8FlowAccumulation", args, output)


def extract_streams(binary: Path, accumulation: Path, output: Path, threshold_cells: int) -> Path:
    args: dict[str, ToolArg] = {
        "flow_accum": accumulation,
        "output": output,
        "threshold": threshold_cells,
        "zero_background": True,
    }
    return run(binary, "ExtractStreams", args, output)


def elevation_above_stream(binary: Path, dem: Path, streams: Path, output: Path) -> Path:
    args: dict[str, ToolArg] = {"dem": dem, "streams": streams, "output": output}
    return run(binary, "ElevationAboveStream", args, output)


def watershed(binary: Path, pointer: Path, pour_points: Path, output: Path) -> Path:
    args: dict[str, ToolArg] = {"d8_pntr": pointer, "pour_pts": pour_points, "output": output}
    return run(binary, "Watershed", args, output)


def streams_to_vector(binary: Path, streams: Path, pointer: Path, output: Path) -> Path:
    args: dict[str, ToolArg] = {"streams": streams, "d8_pntr": pointer, "output": output}
    return run(binary, "RasterStreamsToVector", args, output)


def _read_binary(package: Path) -> bytes:
    # conda package is a zip holding one zstd-compressed tar with the files.
    with zipfile.ZipFile(package) as archive:
        payloads = [
            n for n in archive.namelist() if n.startswith("pkg-") and n.endswith(".tar.zst")
        ]
        if len(payloads) != 1:
            raise ExternalToolError(f"expected one pkg-*.tar.zst in {package.name}")
        with (
            archive.open(payloads[0]) as compressed,
            zstandard.ZstdDecompressor().stream_reader(compressed) as stream,
            tarfile.open(fileobj=stream, mode="r|") as tar,
        ):
            for member in tar:
                if member.name == BINARY_MEMBER and member.isfile():
                    extracted = tar.extractfile(member)
                    if extracted is not None:
                        return extracted.read()
    raise ExternalToolError(f"{BINARY_MEMBER} not found in {package.name}")
