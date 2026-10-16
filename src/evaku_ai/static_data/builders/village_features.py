from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from evaku_ai.core.errors import InvalidInputError

FEATURE_COLUMNS = [
    "village_id",
    "name",
    "district_id",
    "area_km2",
    "basin_share",
    "distance_to_stream_m",
    "population",
    "children_under5",
    "elderly",
    "hand_median",
    "population_hand_le_1m",
    "population_hand_le_3m",
]
M2_PER_KM2 = 1_000_000.0


def build_village_features(processed_dir: Path, epsg: int) -> pd.DataFrame:
    villages = gpd.read_parquet(processed_dir / "villages.parquet")
    population = pd.read_parquet(processed_dir / "village_population.parquet")
    hand = pd.read_parquet(processed_dir / "village_hand.parquet")
    streams = gpd.read_parquet(processed_dir / "streams.parquet").to_crs(epsg=epsg)
    basins = gpd.read_parquet(processed_dir / "river_basins.parquet").to_crs(epsg=epsg)
    if streams.empty or basins.empty:
        raise InvalidInputError("streams and river basins must not be empty")
    for label, frame in (("villages", villages), ("population", population), ("hand", hand)):
        if not frame["village_id"].is_unique:
            raise InvalidInputError(f"village_id must be unique in the {label} table")

    projected = villages.to_crs(epsg=epsg)
    area = projected.geometry.area
    # Distance from the village centroid, not from its nearest edge, to any extracted stream.
    tree = shapely.STRtree(streams.geometry.to_numpy())
    _, distance = tree.query_nearest(
        projected.geometry.centroid.to_numpy(), return_distance=True, all_matches=False
    )
    in_basin = projected.geometry.intersection(basins.geometry.union_all()).area
    table = pd.DataFrame(
        {
            "village_id": villages["village_id"].astype(str).to_numpy(),
            "name": villages["name"].to_numpy(),
            "district_id": villages["district_id"].astype(str).to_numpy(),
            "area_km2": np.round(area.to_numpy() / M2_PER_KM2, 3),
            "basin_share": np.round(in_basin.to_numpy() / area.to_numpy(), 4),
            "distance_to_stream_m": np.round(distance, 1),
        }
    )
    table = table.merge(population.round(1), on="village_id", how="left")
    table = table.merge(hand.round(2), on="village_id", how="left")
    table = table[FEATURE_COLUMNS]
    if len(table) != len(villages) or not table["village_id"].is_unique:
        raise InvalidInputError("village features must have one row per village")
    missing = table.columns[table.isna().any()].tolist()
    if missing:
        raise InvalidInputError(f"village features have missing values in {missing}")
    return table
