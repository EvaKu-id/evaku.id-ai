from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csgraph

from evaku_ai.core.errors import InvalidInputError
from evaku_ai.domain.arrays import BoolArray, FloatArray, IntArray
from evaku_ai.domain.enums import Vehicle
from evaku_ai.domain.models import EvacuationPlan, EvacuationSite
from evaku_ai.domain.rules import EvacuationRules, PassabilityThresholds
from evaku_ai.engine.accessibility import NO_NODE, NodePopulation
from evaku_ai.engine.flooding import depth_from_water_level, edge_status, passable_edges
from evaku_ai.engine.graph import to_csr
from evaku_ai.engine.routing import edge_lookup, path_edges, path_from_predecessors
from evaku_ai.engine.simulation import ScenarioData

M_PER_KM = 1000.0
S_PER_MIN = 60.0
MIN_PER_H = 60.0


@dataclass(frozen=True)
class EvacuationSites:
    facility_ids: IntArray
    nodes: IntArray
    hand: FloatArray
    types: list[str]
    preferred: BoolArray

    def __post_init__(self) -> None:
        n = self.facility_ids.shape[0]
        lengths = (
            self.nodes.shape[0],
            self.hand.shape[0],
            len(self.types),
            self.preferred.shape[0],
        )
        if any(length != n for length in lengths):
            raise InvalidInputError("evacuation site arrays must have the same length")
        if np.unique(self.facility_ids).shape[0] != n:
            raise InvalidInputError("evacuation site facility_ids must be unique")
        if self.preferred.dtype != np.bool_:
            raise InvalidInputError("preferred must be a bool array")


@dataclass(frozen=True)
class Horizon:
    hours: int
    water_level_m: float
    eligible: BoolArray


@dataclass(frozen=True)
class _Reach:
    distance: FloatArray
    predecessor: IntArray
    safe: BoolArray
    usable: BoolArray
    lookup: dict[tuple[int, int], int]


def village_origins(population: NodePopulation, n_villages: int) -> IntArray:
    origins = np.full(n_villages, NO_NODE, dtype=np.int64)
    nodes = np.flatnonzero(population.village != NO_NODE)
    if not nodes.size:
        return origins
    # Sorted so the last node of each village is its most populated one (ties: lowest node_id).
    order = nodes[np.lexsort((-nodes, population.total[nodes], population.village[nodes]))]
    village = population.village[order]
    last = np.append(village[1:] != village[:-1], True)
    origins[village[last]] = order[last]
    return origins


def safe_sites(sites: EvacuationSites, water_level_m: float, rules: EvacuationRules) -> BoolArray:
    allowed = np.isin(np.array(sites.types, dtype=str), [str(t) for t in rules.site_types])
    high = np.nan_to_num(sites.hand, nan=-np.inf) >= water_level_m + rules.safety_margin_m
    return allowed & high


def nearest_safe_sites(
    distance_m: FloatArray, sites: EvacuationSites, safe: BoolArray, rules: EvacuationRules
) -> IntArray:
    reach = distance_m[sites.nodes]
    usable = np.flatnonzero(safe & np.isfinite(reach) & (reach <= rules.max_distance_m))
    # Official BPBD sites come first; then the shortest road distance; facility_id breaks ties.
    order = np.lexsort((sites.facility_ids[usable], reach[usable], ~sites.preferred[usable]))
    return usable[order][: rules.max_sites]


def plan_evacuation(
    data: ScenarioData,
    sites: EvacuationSites,
    river_id: str,
    horizons: Sequence[Horizon],
    rules: EvacuationRules = EvacuationRules(),
    thresholds: PassabilityThresholds = PassabilityThresholds(),
    current_water_level_m: float = 0.0,
) -> list[EvacuationPlan]:
    if river_id not in data.rivers:
        raise InvalidInputError(f"unknown river_id {river_id!r}")
    if not np.isfinite(current_water_level_m) or current_water_level_m < 0:
        raise InvalidInputError(
            f"current water level must be finite and >= 0, got {current_water_level_m}"
        )
    n_villages = len(data.village_ids)
    _check_horizons(horizons, n_villages)
    _check_sites(sites, data.graph.n_nodes)
    origins = village_origins(data.population, n_villages)
    wanted = np.zeros(n_villages, dtype=bool)
    for horizon in horizons:
        wanted |= horizon.eligible
    sources = np.unique(origins[wanted & (origins != NO_NODE)])
    row = {int(node): i for i, node in enumerate(sources)}
    river = data.rivers.index(river_id)
    # People leave under today's road conditions; each horizon says where is still dry then.
    now = _reach(data, sites, river, current_water_level_m, sources, rules, thresholds)
    reach = [
        _reach(data, sites, river, h.water_level_m, sources, rules, thresholds) for h in horizons
    ]

    plans = []
    for h, horizon in enumerate(horizons):
        rows: list[EvacuationSite] = []
        unserved: list[str] = []
        safe = now.safe & reach[h].safe
        for village in np.flatnonzero(horizon.eligible).tolist():
            origin = int(origins[village])
            chosen = (
                nearest_safe_sites(now.distance[row[origin]], sites, safe, rules)
                if origin != NO_NODE
                else np.empty(0, dtype=np.int64)
            )
            if not chosen.size:
                unserved.append(data.village_ids[village])
                continue
            for rank, site in enumerate(chosen.tolist(), start=1):
                rows.append(
                    _site_row(
                        data, sites, horizons, now, reach, row[origin], village, site, rank, rules
                    )
                )
        plans.append(EvacuationPlan(horizon.hours, rows, unserved))
    return plans


def _reach(
    data: ScenarioData,
    sites: EvacuationSites,
    river: int,
    water_level_m: float,
    sources: IntArray,
    rules: EvacuationRules,
    thresholds: PassabilityThresholds,
) -> _Reach:
    graph = data.graph
    depth = depth_from_water_level(graph.hand_min, water_level_m, data.edge_river == river)
    status, _ = edge_status(graph, depth, thresholds)
    usable = passable_edges(status, Vehicle.CAR)
    safe = safe_sites(sites, water_level_m, rules)
    if not sources.size:
        empty = np.empty((0, graph.n_nodes))
        return _Reach(empty, empty.astype(np.int64), safe, usable, {})
    csr = to_csr(graph, usable, weights=graph.length_m)
    distance, predecessor = csgraph.dijkstra(
        csr,
        directed=True,
        indices=sources,
        limit=rules.max_distance_m,
        return_predecessors=True,
    )
    predecessor = np.where(predecessor < 0, NO_NODE, predecessor).astype(np.int64)
    return _Reach(
        np.atleast_2d(distance),
        np.atleast_2d(predecessor),
        safe,
        usable,
        edge_lookup(graph, usable, graph.length_m),
    )


def _site_row(
    data: ScenarioData,
    sites: EvacuationSites,
    horizons: Sequence[Horizon],
    now: _Reach,
    reach: list[_Reach],
    source_row: int,
    village: int,
    site: int,
    rank: int,
    rules: EvacuationRules,
) -> EvacuationSite:
    node = int(sites.nodes[site])
    path = path_from_predecessors(now.predecessor[source_row], node)
    edges = path_edges(path, now.lookup)
    distance = float(now.distance[source_row, node])
    return EvacuationSite(
        village_id=data.village_ids[village],
        rank=rank,
        facility_id=int(sites.facility_ids[site]),
        distance_m=distance,
        walk_minutes=distance / (rules.walk_speed_kmh * M_PER_KM / MIN_PER_H),
        drive_minutes=float(data.graph.travel_s[edges].sum()) / S_PER_MIN,
        leave_before_hour=_first_cut(horizons, reach, edges, site),
        route_edge_ids=tuple(int(data.edge_ids[e]) for e in edges),
    )


def _first_cut(
    horizons: Sequence[Horizon], reach: list[_Reach], edges: list[int], site: int
) -> int | None:
    # The route shown to people is the one that must stay open, not just any way to the site.
    for horizon, state in zip(horizons, reach, strict=True):
        if not (state.usable[edges].all() and state.safe[site]):
            return horizon.hours
    return None


def _check_horizons(horizons: Sequence[Horizon], n_villages: int) -> None:
    hours = [h.hours for h in horizons]
    if not horizons or hours != sorted(set(hours)):
        raise InvalidInputError(
            f"horizons must be non-empty with unique, increasing hours: {hours}"
        )
    for h in horizons:
        if not np.isfinite(h.water_level_m) or h.water_level_m < 0:
            raise InvalidInputError(f"water level at {h.hours} h must be finite and >= 0")
        if h.eligible.dtype != np.bool_ or h.eligible.shape != (n_villages,):
            raise InvalidInputError(f"eligible at {h.hours} h must be a bool array of {n_villages}")


def _check_sites(sites: EvacuationSites, n_nodes: int) -> None:
    if sites.nodes.size and (sites.nodes.min() < 0 or sites.nodes.max() >= n_nodes):
        raise InvalidInputError(f"evacuation site nodes must be in 0..{n_nodes - 1}")
