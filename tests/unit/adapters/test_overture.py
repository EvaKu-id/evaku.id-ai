import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import shapely
from pyarrow import fs

from evaku_ai.adapters.overture import fetch_places
from evaku_ai.core.errors import DownloadError
from evaku_ai.domain.region import OvertureSource

RELEASE = "2026-09-23.0"


def write_release(root):
    folder = root / "release" / RELEASE / "theme=places" / "type=place"
    folder.mkdir(parents=True)
    points = [("a", 98.05, 4.28), ("b", 98.10, 4.30), ("c", 99.50, 4.30)]
    table = pa.table(
        {
            "id": [p[0] for p in points],
            "names": [{"primary": f"Place {p[0]}"} for p in points],
            "basic_category": ["school", "hospital", "school"],
            "confidence": [0.9, 0.8, 0.9],
            "geometry": [shapely.to_wkb(shapely.Point(p[1], p[2])) for p in points],
            "bbox": [{"xmin": p[1], "xmax": p[1], "ymin": p[2], "ymax": p[2]} for p in points],
        }
    )
    pq.write_table(table, folder / "part-0.parquet")


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return OvertureSource(
        release=RELEASE, path="data/raw/places.parquet", license="CDLA Permissive 2.0"
    )


def test_fetch_filters_bbox_and_caches(tmp_path, source):
    write_release(tmp_path / "bucket")
    path = fetch_places(
        source,
        (97.7, 3.8, 98.3, 4.6),
        filesystem=fs.LocalFileSystem(),
        root=str(tmp_path / "bucket"),
    )
    assert pq.read_table(path).column("id").to_pylist() == ["a", "b"]
    again = fetch_places(source, (0, 0, 1, 1), filesystem=fs.LocalFileSystem(), root="/nowhere")
    assert again == path


def test_fetch_failure_leaves_no_file(tmp_path, source):
    with pytest.raises(DownloadError, match=RELEASE):
        fetch_places(
            source,
            (97.7, 3.8, 98.3, 4.6),
            filesystem=fs.LocalFileSystem(),
            root=str(tmp_path / "missing"),
        )
    assert not (tmp_path / "data" / "raw").joinpath("places.parquet").exists()
    assert not (tmp_path / "data" / "raw").joinpath("places.parquet.part").exists()
