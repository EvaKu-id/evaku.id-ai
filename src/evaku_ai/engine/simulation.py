from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np

from evaku_ai.core.errors import InvalidInputError
from evaku_ai.domain.arrays import BoolArray, FloatArray, IntArray, StatusArray
from evaku_ai.domain.enums import RoadCondition, Vehicle, VillageAccessStatus
from evaku_ai.domain.models import AccessSummary, RoadChange, VillageResult
from evaku_ai.domain.rules import IsolationRules, PassabilityThresholds, ScoreWeights
from evaku_ai.engine.accessibility import (
    NO_NODE,
    NodePopulation,
    access_difficulty,
    access_times,
    classify_villages,
    health_access_gap,
)
from evaku_ai.engine.flooding import (
    depth_from_water_level,
    edge_status,
    passable_edges,
)
from evaku_ai.engine.graph import RoadGraph
from evaku_ai.engine.scoring import ScoreResult, priority_scores

M_PER_KM = 1000.0


@dataclass(frozen=True)
class ScenarioData:
    # *_river holds the index into rivers of the drainage each object's HAND refers to.
    graph: RoadGraph
    edge_ids: IntArray
    rivers: list[str]
    edge_river: IntArray
    node_river: IntArray
    node_hand: FloatArray
    population: NodePopulation
    village_ids: list[str]
    hub_nodes: dict[int, int]
    health_nodes: IntArray
    health_hand: FloatArray
    health_river: IntArray

    def __post_init__(self) -> None:
        n_edges, n_nodes = self.graph.n_edges, self.graph.n_nodes
        _check_shape(self.edge_ids, n_edges, "edge_ids")
        _check_shape(self.edge_river, n_edges, "edge_river")
        _check_shape(self.node_river, n_nodes, "node_river")
        _check_shape(self.node_hand, n_nodes, "node_hand")
        _check_shape(self.population.village, n_nodes, "population")
        n_health = self.health_nodes.shape[0]
        _check_shape(self.health_hand, n_health, "health_hand")
        _check_shape(self.health_river, n_health, "health_river")
        if np.unique(self.edge_ids).shape[0] != n_edges:
            raise InvalidInputError("edge_ids must be unique")
        for name, rivers in (
            ("edge_river", self.edge_river),
            ("node_river", self.node_river),
            ("health_river", self.health_river),
        ):
            _check_range(rivers, NO_NODE, len(self.rivers), name)
        _check_range(self.population.village, NO_NODE, len(self.village_ids), "population village")
        _check_range(self.health_nodes, 0, n_nodes, "health_nodes")
        if not self.hub_nodes:
            raise InvalidInputError("at least one logistics hub is required")
        _check_range(np.array(list(self.hub_nodes.values())), 0, n_nodes, "hub_nodes")


@dataclass(frozen=True)
class SimulationResult:
    roads: list[RoadChange]
    villages: list[VillageResult]
    summary: AccessSummary
    scores: ScoreResult
    edge_status: StatusArray = field(compare=False)
    hub_nodes: Mapping[int, int] = field(compare=False)


def run_simulation(
    data: ScenarioData,
    river_id: str,
    water_level_m: float,
    vehicle: Vehicle,
    hub_ids: list[int] | None = None,
    *,
    thresholds: PassabilityThresholds = PassabilityThresholds(),
    isolation: IsolationRules = IsolationRules(),
    weights: ScoreWeights = ScoreWeights(),
) -> SimulationResult:
    if river_id not in data.rivers:
        raise InvalidInputError(f"unknown river_id {river_id!r}")
    river = data.rivers.index(river_id)
    hubs = _hub_nodes(data, hub_ids)
    graph = data.graph

    depth = depth_from_water_level(graph.hand_min, water_level_m, data.edge_river == river)
    status, depth = edge_status(graph, depth, thresholds)
    access = access_times(graph, status, np.array(list(hubs.values()), dtype=np.int64))

    flooded_node = _flooded(data.node_hand, data.node_river == river, water_level_m)
    villages = classify_villages(access, data.population, flooded_node, data.village_ids, isolation)

    n_villages = len(data.village_ids)
    health_flooded = _flooded(data.health_hand, data.health_river == river, water_level_m)
    components = {
        "isolated_population": np.array([v.isolated_population for v in villages], dtype=float),
        "vulnerable_share": np.array([_vulnerable_share(v) for v in villages]),
        "no_health_facility": health_access_gap(
            graph, status, data.health_nodes, health_flooded, data.population, n_villages
        ),
        "access_difficulty": access_difficulty(access, data.population, n_villages, isolation),
    }
    eligible = np.array([v.status is not VillageAccessStatus.REACHABLE for v in villages])
    scores = priority_scores(components, eligible, weights)
    villages = [
        replace(village, priority_score=float(score))
        for village, score in zip(villages, scores.score, strict=True)
    ]

    return SimulationResult(
        roads=_unsafe_roads(data.edge_ids, status, depth),
        villages=villages,
        summary=AccessSummary(
            isolated_villages=int(eligible.sum()),
            isolated_population=sum(v.isolated_population for v in villages),
            cut_road_km=_blocked_km(graph, status, vehicle),
        ),
        scores=scores,
        edge_status=status,
        hub_nodes=hubs,
    )


def _hub_nodes(data: ScenarioData, hub_ids: list[int] | None) -> dict[int, int]:
    if not hub_ids:
        return dict(data.hub_nodes)
    unknown = sorted(set(hub_ids) - set(data.hub_nodes))
    if unknown:
        raise InvalidInputError(f"unknown hub_ids {unknown}")
    return {hub_id: data.hub_nodes[hub_id] for hub_id in sorted(set(hub_ids))}


def _flooded(hand: FloatArray, connected: BoolArray, water_level_m: float) -> BoolArray:
    return depth_from_water_level(hand, water_level_m, connected) > 0


def _vulnerable_share(village: VillageResult) -> float:
    if village.isolated_population == 0:
        return 0.0
    return (village.isolated_under5 + village.isolated_elderly) / village.isolated_population


def _unsafe_roads(
    edge_ids: IntArray, status: StatusArray, depth_cm: FloatArray
) -> list[RoadChange]:
    unsafe = np.flatnonzero(status != RoadCondition.SAFE)
    return [
        RoadChange(
            edge_id=int(edge_ids[i]),
            status=RoadCondition(int(status[i])),
            depth_cm=None if np.isnan(depth_cm[i]) else round(float(depth_cm[i])),
        )
        for i in unsafe
    ]


def _blocked_km(graph: RoadGraph, status: StatusArray, vehicle: Vehicle) -> float:
    blocked = ~passable_edges(status, vehicle)
    return float(graph.length_m[blocked].sum() / M_PER_KM)


def _check_shape(values: IntArray | FloatArray, length: int, name: str) -> None:
    if values.shape != (length,):
        raise InvalidInputError(f"{name} must be a 1-D array of length {length}")


def _check_range(values: IntArray, low: int, high: int, name: str) -> None:
    if not np.issubdtype(values.dtype, np.integer):
        raise InvalidInputError(f"{name} must be an integer array")
    if values.size and (values.min() < low or values.max() >= high):
        raise InvalidInputError(f"{name} index out of range {low}..{high - 1}")
