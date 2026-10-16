import logging
import os
import shutil
import subprocess
from pathlib import Path

from evaku_ai.core.errors import ExternalToolError

log = logging.getLogger(__name__)

TAG = "2.79.0"
TOOLS_DIR = Path("data/tools") / f"tippecanoe-{TAG}"
RUN_TIMEOUT_S = 1800.0
VERSION_TIMEOUT_S = 60.0
STDERR_TAIL = 2000


def find_binary(name: str = "tippecanoe", tools_dir: Path = TOOLS_DIR) -> Path:
    local = tools_dir / name
    if local.is_file() and os.access(local, os.X_OK):
        return local
    found = shutil.which(name)
    if found is None:
        raise ExternalToolError(f"{name} not found; run `make tippecanoe`")
    return Path(found)


def version(binary: Path) -> str:
    try:
        # Argument list without a shell; the binary is built from a pinned commit.
        done = subprocess.run(  # noqa: S603
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            timeout=VERSION_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ExternalToolError(f"{binary.name} --version timed out") from exc
    if done.returncode != 0:
        raise ExternalToolError(f"{binary.name} --version failed ({done.returncode})")
    # tippecanoe prints its version to stderr.
    text = (done.stderr or done.stdout).strip()
    return text.splitlines()[0] if text else "unknown"


def run(args: list[str], output: Path) -> Path:
    log.info("tippecanoe %s", output.name)
    try:
        done = subprocess.run(  # noqa: S603
            args, capture_output=True, text=True, timeout=RUN_TIMEOUT_S, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise ExternalToolError(
            f"tippecanoe {output.name} timed out after {RUN_TIMEOUT_S:.0f} s"
        ) from exc
    if done.returncode != 0 or not output.is_file():
        tail = (done.stderr or done.stdout)[-STDERR_TAIL:]
        raise ExternalToolError(f"tippecanoe {output.name} failed ({done.returncode}): {tail}")
    return output
