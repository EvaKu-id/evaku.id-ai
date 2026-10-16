from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, box

from evaku_ai.adapters.bundle import write_bundle
from evaku_ai.adapters.download import file_sha256
from evaku_ai.engine.synthetic import synthetic_scenario
from evaku_ai.static_data.builders.bundle import INPUTS
from evaku_ai.static_data.release import GEO_FILES, META_FILES
from tests.fixtures.evacuation import sample_sites

UTM = 32647


def _utm(geometries, **columns):
    return gpd.GeoDataFrame(columns, geometry=geometries, crs=UTM).to_crs(4326)


def write_village_inputs(out: Path, population=None, hand=None, streams: bool = True) -> Path:
    _utm(
        [box(400000, 460000, 402000, 462000), box(402000, 460000, 404000, 462000)],
        village_id=["1114000001", "1114000002"],
        name=["Alpha", "Beta"],
        district_id=["1114000", "1114000"],
    ).to_parquet(out / "villages.parquet")
    if population is None:
        population = pd.DataFrame(
            {
                "village_id": ["1114000001", "1114000002"],
                "population": [1000.04, 250.0],
                "children_under5": [90.0, 20.0],
                "elderly": [95.0, 25.0],
            }
        )
    population.to_parquet(out / "village_population.parquet")
    if hand is None:
        hand = pd.DataFrame(
            {
                "village_id": ["1114000001", "1114000002"],
                "hand_median": [1.234, 6.0],
                "population_hand_le_1m": [100.0, 0.0],
                "population_hand_le_3m": [400.0, 10.0],
            }
        )
    hand.to_parquet(out / "village_hand.parquet")
    lines = [LineString([(399500, 458000), (399500, 464000)])] if streams else []
    _utm(lines, id=range(len(lines))).to_parquet(out / "streams.parquet")
    _utm([box(400000, 459000, 401000, 463000)], river_id=["tamiang"]).to_parquet(
        out / "river_basins.parquet"
    )
    return out


def write_processed(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    write_village_inputs(out)
    filler = pd.DataFrame({"id": [1, 2, 3]})
    for name in GEO_FILES:
        if not (out / name).exists() and name != "hand.tif":
            filler.to_parquet(out / name)
    (out / "hand.tif").write_bytes(b"II*\x00fake-geotiff")
    for name in META_FILES:
        (out / name).write_text("{}", encoding="utf-8")
    return out


def make_release_bundle(processed_dir: Path, static_dir: Path, version: str) -> Path:
    sources = {name: file_sha256(processed_dir / name) for name in INPUTS}
    static_dir.mkdir(parents=True, exist_ok=True)
    return write_bundle(
        synthetic_scenario(),
        sample_sites(),
        static_dir,
        data_version=version,
        region_id="aceh-tamiang",
        sources=sources,
    )
