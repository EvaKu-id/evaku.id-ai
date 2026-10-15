from collections.abc import Sequence
from pathlib import Path

from shapely.geometry.base import BaseGeometry

from evaku_ai.adapters.raster import read_depth_raster
from evaku_ai.domain.arrays import FloatArray, StatusArray
from evaku_ai.domain.rules import HandRules, PassabilityThresholds
from evaku_ai.engine.flooding import depth_from_observation, edge_status
from evaku_ai.engine.graph import RoadGraph
from evaku_ai.geo.raster import max_along_lines


def run(mode: object) -> object:
    raise NotImplementedError


def road_status_from_depth(
    raster_path: Path,
    edge_lines: Sequence[BaseGeometry],
    graph: RoadGraph,
    epsg: int,
    thresholds: PassabilityThresholds = PassabilityThresholds(),
    step_m: float = HandRules().sample_step_m,
) -> tuple[StatusArray, FloatArray]:
    # Unlike HAND, stream crossings are not skipped: water seen over a culvert is real flooding.
    depth = read_depth_raster(raster_path, epsg)
    depth_m = max_along_lines(depth.values, depth.transform, None, edge_lines, step_m)
    return edge_status(graph, depth_from_observation(depth_m), thresholds)
