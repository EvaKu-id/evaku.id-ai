import itertools
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from evaku_ai.core.errors import InvalidInputError
from evaku_ai.domain.arrays import BoolArray, FloatArray, IntArray, StatusArray
from evaku_ai.domain.enums import RouteMode, Vehicle
from evaku_ai.domain.models import BoatPoint, VillageRoute
from evaku_ai.domain.rules import RouteRules
from evaku_ai.engine.accessibility import NO_NODE, travel_time_from_hubs, village_origins
from evaku_ai.engine.flooding import passable_edges
from evaku_ai.engine.graph import RoadGeometry, RoadGraph, to_csr
from evaku_ai.engine.simulation import ScenarioData

M_PER_KM = 1000.0
S_PER_MIN = 60.0
MIN_PER_H = 60.0


def path_from_predecessors(predecessors: IntArray, target: int) -> list[int]:
    path = [target]
    node = int(predecessors[target])
    while node != NO_NODE:
        if len(path) > predecessors.shape[0]:
            raise InvalidInputError("predecessor chain loops")
        path.append(node)
        node = int(predecessors[node])
    path.reverse()
    return path


def edge_lookup(
    graph: RoadGraph, passable: BoolArray, weights: FloatArray
) -> dict[tuple[int, int], int]:
    best: dict[tuple[int, int], int] = {}
    for edge in np.flatnonzero(passable).tolist():
        u, v = int(graph.edge_u[edge]), int(graph.edge_v[edge])
        key = (min(u, v), max(u, v))
        if key not in best or weights[edge] < weights[best[key]]:
            best[key] = edge
    return best


def path_edges(nodes: list[int], lookup: dict[tuple[int, int], int]) -> list[int]:
    edges = []
    for a, b in itertools.pairwise(nodes):
        key = (min(a, b), max(a, b))
        if key not in lookup:
            raise InvalidInputError(f"no usable edge between nodes {a} and {b}")
        edges.append(lookup[key])
    return edges


def route_coordinates(
    graph: RoadGraph, geometry: RoadGeometry, path: list[int], edges: list[int]
) -> tuple[tuple[float, float], ...]:
    if not edges:
        node = path[0]
        point = (float(geometry.node_lon[node]), float(geometry.node_lat[node]))
        return (point, point)
    parts: list[FloatArray] = []
    for (a, _), edge in zip(itertools.pairwise(path), edges, strict=True):
        start, end = geometry.edge_offsets[edge], geometry.edge_offsets[edge + 1]
        coords = geometry.edge_coords[start:end]
        if graph.edge_u[edge] == a:
            part = coords
        elif graph.edge_v[edge] == a:
            part = coords[::-1]
        else:
            raise InvalidInputError(f"edge {edge} does not start or end at node {a}")
        parts.append(part if not parts else part[1:])
    joined = np.concatenate(parts)
    return tuple((float(lon), float(lat)) for lon, lat in joined)


def plan_routes(
    data: ScenarioData,
    geometry: RoadGeometry,
    status: StatusArray,
    vehicle: Vehicle,
    villages: Sequence[int],
    hub_nodes: Mapping[int, int],
    rules: RouteRules = RouteRules(),
) -> list[VillageRoute]:
    graph = data.graph
    _check_inputs(data, status, villages, hub_nodes)
    passable = passable_edges(status, vehicle)
    sources = np.array(sorted(set(hub_nodes.values())), dtype=np.int64)
    reach = travel_time_from_hubs(to_csr(graph, passable), sources)
    lookup = edge_lookup(graph, passable, graph.travel_s)
    origins = village_origins(data.population, len(data.village_ids))
    hub_of_node: dict[int, int] = {}
    for hub_id, node in sorted(hub_nodes.items()):
        hub_of_node.setdefault(node, hub_id)
    reachable = np.flatnonzero(reach.reachable)
    tree: cKDTree | None = None
    boat_m_per_min = rules.boat_speed_kmh * M_PER_KM / MIN_PER_H

    routes = []
    for village in villages:
        village_id = data.village_ids[village]
        target = int(origins[village])
        if target == NO_NODE:
            raise InvalidInputError(f"village {village_id} has no road node")
        boat: BoatPoint | None = None
        water_m = 0.0
        end = target
        if not reach.reachable[target]:
            if tree is None:
                tree = cKDTree(np.column_stack([graph.node_x[reachable], graph.node_y[reachable]]))
            distance, index = tree.query((graph.node_x[target], graph.node_y[target]))
            water_m = float(distance)
            end = int(reachable[int(index)])
            boat = BoatPoint(
                lon=float(geometry.node_lon[end]),
                lat=float(geometry.node_lat[end]),
                water_distance_m=water_m,
            )
        path = path_from_predecessors(reach.predecessor, end)
        edges = path_edges(path, lookup)
        land_m = float(graph.length_m[edges].sum())
        land_s = float(graph.travel_s[edges].sum())
        routes.append(
            VillageRoute(
                village_id=village_id,
                hub_id=hub_of_node[int(reach.hub[end])],
                mode=RouteMode.LAND if boat is None else RouteMode.LAND_BOAT,
                coordinates=route_coordinates(graph, geometry, path, edges),
                edge_ids=tuple(int(data.edge_ids[e]) for e in edges),
                distance_m=land_m + water_m,
                minutes=land_s / S_PER_MIN + water_m / boat_m_per_min,
                boat_point=boat,
            )
        )
    return routes


def _check_inputs(
    data: ScenarioData,
    status: StatusArray,
    villages: Sequence[int],
    hub_nodes: Mapping[int, int],
) -> None:
    graph = data.graph
    if np.shape(status) != (graph.n_edges,):
        raise InvalidInputError(f"status must have length {graph.n_edges}")
    if not hub_nodes:
        raise InvalidInputError("at least one hub is required")
    if any(not 0 <= node < graph.n_nodes for node in hub_nodes.values()):
        raise InvalidInputError(f"hub nodes must be in 0..{graph.n_nodes - 1}")
    n_villages = len(data.village_ids)
    if any(not 0 <= village < n_villages for village in villages):
        raise InvalidInputError(f"village index must be in 0..{n_villages - 1}")
