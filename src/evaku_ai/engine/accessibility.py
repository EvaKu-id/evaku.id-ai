from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from evaku_ai.core.errors import InvalidInputError
from evaku_ai.domain.arrays import BoolArray, FloatArray, IntArray, StatusArray
from evaku_ai.domain.enums import NodeImpact, Vehicle, VillageAccessStatus
from evaku_ai.domain.models import VillageResult
from evaku_ai.domain.rules import IsolationRules
from evaku_ai.engine.flooding import passable_edges
from evaku_ai.engine.graph import RoadGraph, to_csr

NO_NODE = -1
_SCIPY_NO_NODE = -9999


@dataclass(frozen=True)
class Reachability:
    time_s: FloatArray
    hub: IntArray
    predecessor: IntArray

    @property
    def reachable(self) -> BoolArray:
        return np.isfinite(self.time_s)


@dataclass(frozen=True)
class AccessTimes:
    normal: Reachability
    flood: dict[Vehicle, Reachability]


@dataclass(frozen=True)
class NodePopulation:
    # village holds the village index of each node, NO_NODE when the node lies outside all villages.
    village: IntArray
    total: FloatArray
    under5: FloatArray
    elderly: FloatArray

    def __post_init__(self) -> None:
        counts = (self.total, self.under5, self.elderly)
        if self.village.ndim != 1 or any(c.shape != self.village.shape for c in counts):
            raise InvalidInputError("population arrays must be 1-D with the same length")
        if not np.issubdtype(self.village.dtype, np.integer):
            raise InvalidInputError("village must be an integer array")
        if self.village.size and self.village.min() < NO_NODE:
            raise InvalidInputError(f"village index must be >= {NO_NODE}")
        if not all(np.all(np.isfinite(c) & (c >= 0)) for c in counts):
            raise InvalidInputError("population counts must be finite and >= 0")


@dataclass(frozen=True)
class FacilityAccess:
    flooded: BoolArray
    reachable: dict[Vehicle, BoolArray]


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


def travel_time_from_hubs(csr: sparse.csr_matrix, hub_nodes: IntArray) -> Reachability:
    hubs = _validate_hubs(hub_nodes, csr.shape[0])
    # to_csr builds a symmetric matrix, so directed search skips scipy's symmetrizing copy.
    # With min_only, return_predecessors also returns the source hub of each node.
    time_s, predecessor, hub = csgraph.dijkstra(
        csr, directed=True, indices=hubs, min_only=True, return_predecessors=True
    )
    return Reachability(
        time_s=np.asarray(time_s, dtype=np.float64),
        hub=_to_node_ids(hub),
        predecessor=_to_node_ids(predecessor),
    )


def access_times(graph: RoadGraph, status: StatusArray, hub_nodes: IntArray) -> AccessTimes:
    if np.shape(status) != (graph.n_edges,):
        raise InvalidInputError(f"status must have length {graph.n_edges}")
    normal = travel_time_from_hubs(to_csr(graph), hub_nodes)
    flood = {
        vehicle: travel_time_from_hubs(to_csr(graph, passable_edges(status, vehicle)), hub_nodes)
        for vehicle in Vehicle
    }
    return AccessTimes(normal=normal, flood=flood)


def classify_villages(
    access: AccessTimes,
    population: NodePopulation,
    flooded_node: BoolArray,
    village_ids: list[str],
    rules: IsolationRules = IsolationRules(),
) -> list[VillageResult]:
    n_nodes = access.normal.time_s.shape[0]
    n_villages = len(village_ids)
    _check_population(population, n_nodes, n_villages)
    _check_bool_mask(flooded_node, (n_nodes,), "flooded_node")

    impact = node_impact(access, rules)
    cut = impact == NodeImpact.CUT
    affected = impact != NodeImpact.NORMAL

    def per_village(weights: FloatArray) -> FloatArray:
        return _per_village(population, weights, n_villages)

    total = per_village(population.total)
    isolated = per_village(population.total * cut)
    affected_pop = per_village(population.total * affected)
    flooded = per_village(population.total * flooded_node)
    under5 = per_village(population.under5 * cut)
    elderly = per_village(population.elderly * cut)

    return [
        VillageResult(
            village_id=village_id,
            status=_village_status(total[i], isolated[i], affected_pop[i], rules),
            isolated_population=round(isolated[i]),
            flooded_population=round(flooded[i]),
            isolated_under5=round(under5[i]),
            isolated_elderly=round(elderly[i]),
            priority_score=None,
        )
        for i, village_id in enumerate(village_ids)
    ]


def facility_access(
    access: AccessTimes, facility_nodes: IntArray, facility_flooded: BoolArray
) -> FacilityAccess:
    nodes = _node_indices(facility_nodes, access.normal.time_s.shape[0], "facility_nodes")
    _check_bool_mask(facility_flooded, nodes.shape, "facility_flooded")
    return FacilityAccess(
        flooded=facility_flooded.copy(),
        reachable={vehicle: reach.reachable[nodes] for vehicle, reach in access.flood.items()},
    )


def node_impact(access: AccessTimes, rules: IsolationRules = IsolationRules()) -> IntArray:
    normal = access.normal.time_s
    car = access.flood[Vehicle.CAR].time_s
    truck = access.flood[Vehicle.TRUCK].time_s
    # Nodes already unreachable on dry roads are a data gap, not flood isolation.
    baseline = np.isfinite(normal)
    impact = np.full(normal.shape, NodeImpact.NORMAL, dtype=np.int64)
    impact[baseline & np.isfinite(car) & (car > rules.delay_factor * normal)] = NodeImpact.DELAYED
    impact[baseline & np.isfinite(truck) & ~np.isfinite(car)] = NodeImpact.TRUCK_ONLY
    impact[baseline & ~np.isfinite(truck)] = NodeImpact.CUT
    return impact


def access_difficulty(
    access: AccessTimes,
    population: NodePopulation,
    n_villages: int,
    rules: IsolationRules = IsolationRules(),
) -> FloatArray:
    _check_population(population, access.normal.time_s.shape[0], n_villages)
    impact = node_impact(access, rules)
    total = _per_village(population, population.total, n_villages)
    weighted = _per_village(population, population.total * impact, n_villages)
    return _share(weighted, total * NodeImpact.CUT)


def health_access_gap(
    graph: RoadGraph,
    status: StatusArray,
    facility_nodes: IntArray,
    facility_flooded: BoolArray,
    population: NodePopulation,
    n_villages: int,
) -> FloatArray:
    nodes = _node_indices(facility_nodes, graph.n_nodes, "facility_nodes")
    _check_bool_mask(facility_flooded, nodes.shape, "facility_flooded")
    _check_population(population, graph.n_nodes, n_villages)
    total = _per_village(population, population.total, n_villages)
    working = nodes[~facility_flooded]
    if working.size == 0:
        return (total > 0).astype(np.float64)
    reach = travel_time_from_hubs(to_csr(graph, passable_edges(status, Vehicle.TRUCK)), working)
    without = _per_village(population, population.total * ~reach.reachable, n_villages)
    return _share(without, total)


def _check_population(population: NodePopulation, n_nodes: int, n_villages: int) -> None:
    if population.village.shape != (n_nodes,):
        raise InvalidInputError(f"population arrays must have length {n_nodes}")
    if population.village.size and population.village.max() >= n_villages:
        raise InvalidInputError(f"village index out of range 0..{n_villages - 1}")


def _per_village(population: NodePopulation, weights: FloatArray, n_villages: int) -> FloatArray:
    inside = population.village >= 0
    counts = np.bincount(population.village[inside], weights=weights[inside], minlength=n_villages)
    return counts.astype(np.float64)


def _share(part: FloatArray, whole: FloatArray) -> FloatArray:
    return np.divide(part, whole, out=np.zeros_like(part), where=whole > 0)


def _village_status(
    total: float, isolated: float, affected: float, rules: IsolationRules
) -> VillageAccessStatus:
    if total <= 0:
        return VillageAccessStatus.REACHABLE
    if isolated / total > rules.fully_share:
        return VillageAccessStatus.FULLY_ISOLATED
    if affected / total >= rules.partial_share:
        return VillageAccessStatus.PARTIALLY_ISOLATED
    return VillageAccessStatus.REACHABLE


def _validate_hubs(hub_nodes: IntArray, n_nodes: int) -> IntArray:
    hubs = _node_indices(hub_nodes, n_nodes, "hub_nodes")
    if hubs.size == 0:
        raise InvalidInputError("hub_nodes must be non-empty")
    return np.unique(hubs)


def _node_indices(nodes: IntArray, n_nodes: int, name: str) -> IntArray:
    ids = np.asarray(nodes)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise InvalidInputError(f"{name} must be a 1-D integer array")
    if ids.size and (ids.min() < 0 or ids.max() >= n_nodes):
        raise InvalidInputError(f"{name} index out of range 0..{n_nodes - 1}")
    return ids.astype(np.int64)


def _check_bool_mask(mask: BoolArray, shape: tuple[int, ...], name: str) -> None:
    if mask.dtype != np.bool_ or mask.shape != shape:
        raise InvalidInputError(f"{name} must be a bool array of shape {shape}")


def _to_node_ids(nodes: IntArray) -> IntArray:
    ids = np.asarray(nodes, dtype=np.int64)
    ids[ids == _SCIPY_NO_NODE] = NO_NODE
    return ids
