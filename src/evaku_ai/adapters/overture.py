import logging
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pyarrow import fs

from evaku_ai.core.errors import DownloadError
from evaku_ai.domain.region import OvertureSource

log = logging.getLogger(__name__)

OVERTURE_BUCKET = "overturemaps-us-west-2"
OVERTURE_REGION = "us-west-2"
PLACE_COLUMNS = ["id", "names", "basic_category", "confidence", "geometry"]
CONNECT_TIMEOUT_S = 30.0
REQUEST_TIMEOUT_S = 120.0


def fetch_places(
    source: OvertureSource,
    bbox: tuple[float, float, float, float],
    *,
    filesystem: fs.FileSystem | None = None,
    root: str = OVERTURE_BUCKET,
) -> Path:
    path = source.path
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    remote = filesystem or fs.S3FileSystem(
        anonymous=True,
        region=OVERTURE_REGION,
        connect_timeout=CONNECT_TIMEOUT_S,
        request_timeout=REQUEST_TIMEOUT_S,
    )
    prefix = f"{root}/release/{source.release}/theme=places/type=place/"
    log.info("reading Overture places %s for bbox %s", source.release, bbox)
    min_x, min_y, max_x, max_y = bbox
    inside = (
        (pc.field("bbox", "xmin") >= min_x)
        & (pc.field("bbox", "xmax") <= max_x)
        & (pc.field("bbox", "ymin") >= min_y)
        & (pc.field("bbox", "ymax") <= max_y)
    )
    try:
        table = ds.dataset(prefix, filesystem=remote, format="parquet").to_table(
            filter=inside, columns=PLACE_COLUMNS
        )
        pq.write_table(table, part)
        part.replace(path)
    except OSError as exc:
        raise DownloadError(f"reading Overture places {source.release} failed: {exc}") from exc
    finally:
        part.unlink(missing_ok=True)
    return path
