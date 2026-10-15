from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from evaku_ai.adapters.bundle import check_version, write_bundle
from evaku_ai.adapters.download import file_sha256
from evaku_ai.adapters.region import load_region
from evaku_ai.core.errors import InvalidInputError
from evaku_ai.domain.arrays import IntArray
from evaku_ai.domain.enums import FacilityType, LocationSource
from evaku_ai.domain.region import RegionConfig
from evaku_ai.engine.accessibility import NO_NODE, NodePopulation
from evaku_ai.engine.evacuation import EvacuationSites
from evaku_ai.engine.simulation import ScenarioData
from evaku_ai.static_data.builders.graph import to_road_graph

log = logging.getLogger(__name__)

HEALTH_TYPES = (FacilityType.HEALTH_CENTER.value, FacilityType.HOSPITAL.value)
INPUTS = (
    "road_nodes.parquet",
    "road_segments.parquet",
    "edge_hand.parquet",
    "node_hand.parquet",
    "node_population.parquet",
    "villages.parquet",
    "logistics_hubs.parquet",
    "facilities.parquet",
    "facility_hand.parquet",
)


def assemble_scenario(processed_dir: Path, region: RegionConfig) -> ScenarioData:
    tables = _tables(processed_dir)
    nodes, segments = tables["road_nodes"], tables["road_segments"]
    edge_hand = _aligned(tables["edge_hand"], "edge_id", segments["edge_id"])
    node_hand = _aligned(tables["node_hand"], "node_id", nodes["node_id"])
    people = _aligned(tables["node_population"], "node_id", nodes["node_id"])
    rivers = [river.river_id for river in region.rivers]
    village_ids = [str(v) for v in tables["villages"]["village_id"]]

    hubs = tables["logistics_hubs"]
    hubs = hubs[hubs["is_active"].astype(bool)]
    health = tables["facilities"].merge(tables["facility_hand"], on="facility_id", how="left")
    health = health[health["type"].isin(HEALTH_TYPES)]

    return ScenarioData(
        graph=replace(
            to_road_graph(nodes, segments), hand_min=edge_hand["hand_min"].to_numpy(dtype=float)
        ),
        edge_ids=segments["edge_id"].to_numpy(dtype=np.int64),
        rivers=rivers,
        edge_river=_river_index(edge_hand["river_id"], rivers),
        node_river=_river_index(node_hand["river_id"], rivers),
        node_hand=node_hand["hand"].to_numpy(dtype=float),
        population=NodePopulation(
            village=_village_index(people["village_id"], village_ids),
            total=people["population"].to_numpy(dtype=float),
            under5=people["under5"].to_numpy(dtype=float),
            elderly=people["elderly"].to_numpy(dtype=float),
        ),
        village_ids=village_ids,
        hub_nodes=dict(
            zip(
                hubs["hub_id"].astype(int).tolist(),
                hubs["node_id"].astype(int).tolist(),
                strict=True,
            )
        ),
        health_nodes=health["node_id"].to_numpy(dtype=np.int64),
        health_hand=health["hand"].to_numpy(dtype=float),
        health_river=_river_index(health["river_id"], rivers),
    )


def assemble_sites(processed_dir: Path) -> EvacuationSites:
    tables = _tables(processed_dir)
    sites = tables["facilities"].merge(tables["facility_hand"], on="facility_id", how="left")
    return EvacuationSites(
        facility_ids=sites["facility_id"].to_numpy(dtype=np.int64),
        nodes=sites["node_id"].to_numpy(dtype=np.int64),
        hand=sites["hand"].to_numpy(dtype=float),
        types=[str(t) for t in sites["type"]],
        preferred=(sites["source"] == LocationSource.BPBD.value).to_numpy(dtype=bool),
    )


def build(
    region_path: Path,
    version: str,
    processed_dir: Path = Path("data/processed"),
    static_dir: Path = Path("data/static"),
) -> Path:
    check_version(version)
    region = load_region(region_path)
    data = assemble_scenario(processed_dir, region)
    sites = assemble_sites(processed_dir)
    sources = {name: file_sha256(processed_dir / name) for name in INPUTS}
    out = write_bundle(
        data, sites, static_dir, data_version=version, region_id=region.region_id, sources=sources
    )
    log.info(
        "bundle %s: %d edges, %d nodes, %d villages, %d hubs",
        version,
        data.graph.n_edges,
        data.graph.n_nodes,
        len(data.village_ids),
        len(data.hub_nodes),
    )
    return out


def _tables(processed_dir: Path) -> dict[str, pd.DataFrame]:
    missing = [name for name in INPUTS if not (processed_dir / name).is_file()]
    if missing:
        raise InvalidInputError(f"missing {missing}; run the static data builders first")
    geo = {"road_nodes", "road_segments", "villages", "logistics_hubs", "facilities"}
    tables: dict[str, pd.DataFrame] = {}
    for name in INPUTS:
        stem = name.removesuffix(".parquet")
        path = processed_dir / name
        tables[stem] = gpd.read_parquet(path) if stem in geo else pd.read_parquet(path)
    return tables


def _aligned(table: pd.DataFrame, key: str, order: pd.Series[int]) -> pd.DataFrame:
    if not np.array_equal(table[key].to_numpy(), order.to_numpy()):
        raise InvalidInputError(f"{key} order in a HAND or population table differs from the graph")
    return table


def _river_index(values: pd.Series[str], rivers: list[str]) -> IntArray:
    unknown = sorted({str(v) for v in values.dropna()} - set(rivers))
    if unknown:
        raise InvalidInputError(f"river_id {unknown} is not in the region config")
    index = {river: i for i, river in enumerate(rivers)}
    return np.array(
        [index.get(v, NO_NODE) if isinstance(v, str) else NO_NODE for v in values], dtype=np.int64
    )


def _village_index(values: pd.Series[str], village_ids: list[str]) -> IntArray:
    index = {village: i for i, village in enumerate(village_ids)}
    unknown = sorted({str(v) for v in values.dropna()} - set(index))
    if unknown:
        raise InvalidInputError(f"village_id {unknown[:5]} is not in villages.parquet")
    return np.array(
        [index.get(v, NO_NODE) if isinstance(v, str) else NO_NODE for v in values], dtype=np.int64
    )
