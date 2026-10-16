from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from evaku_ai.core.errors import InvalidInputError
from evaku_ai.engine.accessibility import NodePopulation
from evaku_ai.engine.evacuation import EvacuationSites
from evaku_ai.engine.graph import RoadGeometry, RoadGraph, check_geometry
from evaku_ai.engine.simulation import ScenarioData

BUNDLE_FORMAT = 3

Array = npt.NDArray[np.int64 | np.float64 | np.bool_ | np.str_]
Arrays = Mapping[str, Array]

INT_KEYS = (
    "edge_u",
    "edge_v",
    "edge_ids",
    "edge_river",
    "node_river",
    "population_village",
    "hub_ids",
    "hub_nodes",
    "health_nodes",
    "health_river",
    "site_facility_ids",
    "site_nodes",
    "edge_coord_offsets",
)
FLOAT_KEYS = (
    "travel_s",
    "length_m",
    "hand_min",
    "node_x",
    "node_y",
    "node_hand",
    "population_total",
    "population_under5",
    "population_elderly",
    "health_hand",
    "site_hand",
    "node_lon",
    "node_lat",
    "edge_coords",
)
BOOL_KEYS = ("is_bridge", "site_preferred")
STR_KEYS = ("rivers", "village_ids", "site_types")
KEYS = frozenset(INT_KEYS + FLOAT_KEYS + BOOL_KEYS + STR_KEYS)


def bundle_to_arrays(
    data: ScenarioData, sites: EvacuationSites, geometry: RoadGeometry
) -> dict[str, Array]:
    graph, population = data.graph, data.population
    return {
        "edge_u": graph.edge_u,
        "edge_v": graph.edge_v,
        "travel_s": graph.travel_s,
        "length_m": graph.length_m,
        "hand_min": graph.hand_min,
        "is_bridge": graph.is_bridge,
        "node_x": graph.node_x,
        "node_y": graph.node_y,
        "edge_ids": data.edge_ids,
        "rivers": np.array(data.rivers, dtype=str),
        "edge_river": data.edge_river,
        "node_river": data.node_river,
        "node_hand": data.node_hand,
        "population_village": population.village,
        "population_total": population.total,
        "population_under5": population.under5,
        "population_elderly": population.elderly,
        "village_ids": np.array(data.village_ids, dtype=str),
        "hub_ids": np.array(list(data.hub_nodes), dtype=np.int64),
        "hub_nodes": np.array(list(data.hub_nodes.values()), dtype=np.int64),
        "health_nodes": data.health_nodes,
        "health_hand": data.health_hand,
        "health_river": data.health_river,
        "site_facility_ids": sites.facility_ids,
        "site_nodes": sites.nodes,
        "site_hand": sites.hand,
        "site_types": np.array(sites.types, dtype=str),
        "site_preferred": sites.preferred,
        "node_lon": geometry.node_lon,
        "node_lat": geometry.node_lat,
        "edge_coord_offsets": geometry.edge_offsets,
        "edge_coords": geometry.edge_coords,
    }


def bundle_from_arrays(arrays: Arrays) -> tuple[ScenarioData, EvacuationSites, RoadGeometry]:
    missing, extra = KEYS - set(arrays), set(arrays) - KEYS
    if missing or extra:
        raise InvalidInputError(
            f"bundle keys differ: missing {sorted(missing)}, extra {sorted(extra)}"
        )
    ints = {key: _ints(arrays[key], key) for key in INT_KEYS}
    floats = {key: np.asarray(arrays[key], dtype=np.float64) for key in FLOAT_KEYS}
    texts = {key: [str(v) for v in _texts(arrays[key], key)] for key in STR_KEYS}
    if ints["hub_ids"].shape != ints["hub_nodes"].shape:
        raise InvalidInputError("hub_ids and hub_nodes must have the same length")
    data = ScenarioData(
        graph=RoadGraph(
            edge_u=ints["edge_u"],
            edge_v=ints["edge_v"],
            travel_s=floats["travel_s"],
            length_m=floats["length_m"],
            hand_min=floats["hand_min"],
            is_bridge=_bools(arrays["is_bridge"], "is_bridge"),
            node_x=floats["node_x"],
            node_y=floats["node_y"],
        ),
        edge_ids=ints["edge_ids"],
        rivers=texts["rivers"],
        edge_river=ints["edge_river"],
        node_river=ints["node_river"],
        node_hand=floats["node_hand"],
        population=NodePopulation(
            village=ints["population_village"],
            total=floats["population_total"],
            under5=floats["population_under5"],
            elderly=floats["population_elderly"],
        ),
        village_ids=texts["village_ids"],
        hub_nodes=dict(zip(ints["hub_ids"].tolist(), ints["hub_nodes"].tolist(), strict=True)),
        health_nodes=ints["health_nodes"],
        health_hand=floats["health_hand"],
        health_river=ints["health_river"],
    )
    sites = EvacuationSites(
        facility_ids=ints["site_facility_ids"],
        nodes=ints["site_nodes"],
        hand=floats["site_hand"],
        types=texts["site_types"],
        preferred=_bools(arrays["site_preferred"], "site_preferred"),
    )
    geometry = RoadGeometry(
        node_lon=floats["node_lon"],
        node_lat=floats["node_lat"],
        edge_offsets=ints["edge_coord_offsets"],
        edge_coords=floats["edge_coords"],
    )
    check_geometry(data.graph, geometry)
    return data, sites, geometry


def _ints(values: Array, key: str) -> npt.NDArray[np.int64]:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer):
        raise InvalidInputError(f"{key} must hold integers, got {array.dtype}")
    return array.astype(np.int64)


def _bools(values: Array, key: str) -> npt.NDArray[np.bool_]:
    array = np.asarray(values)
    if array.dtype != np.bool_:
        raise InvalidInputError(f"{key} must hold booleans, got {array.dtype}")
    return array.astype(np.bool_)


def _texts(values: Array, key: str) -> npt.NDArray[np.str_]:
    array = np.asarray(values)
    if array.dtype.kind != "U":
        raise InvalidInputError(f"{key} must hold text, got {array.dtype}")
    return array.astype(np.str_)
