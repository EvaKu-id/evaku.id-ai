import copy

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import yaml
from shapely.geometry import LineString, Point, box

from evaku_ai.adapters.bundle import read_bundle
from evaku_ai.adapters.region import load_region
from evaku_ai.core.errors import InvalidInputError
from evaku_ai.engine.accessibility import NO_NODE
from evaku_ai.static_data.builders import bundle as builder
from tests.fixtures.region import VALID

UTM = "EPSG:32647"


def _gdf(frame, geometry):
    return gpd.GeoDataFrame(frame, geometry=geometry, crs=UTM)


def _write_inputs(out):
    out.mkdir(parents=True, exist_ok=True)
    xy = [(0.0, 0.0), (100.0, 0.0), (130.0, 0.0), (230.0, 0.0)]
    points = [Point(p) for p in xy]
    _gdf(
        {"node_id": range(4), "x_m": [p[0] for p in xy], "y_m": [p[1] for p in xy]}, points
    ).to_parquet(out / "road_nodes.parquet")
    _gdf(
        {
            "edge_id": range(3),
            "u": [0, 1, 2],
            "v": [1, 2, 3],
            "length_m": [100.0, 30.0, 100.0],
            "travel_s": [10.0, 3.0, 10.0],
            "is_bridge": [False, True, False],
        },
        [LineString([xy[i], xy[i + 1]]) for i in range(3)],
    ).to_parquet(out / "road_segments.parquet")
    pd.DataFrame(
        {
            "edge_id": range(3),
            "hand_min": [1.0, 0.0, 4.0],
            "river_id": ["tamiang", "tamiang", None],
            "crossings": [0, 1, 0],
        }
    ).to_parquet(out / "edge_hand.parquet")
    pd.DataFrame(
        {
            "node_id": range(4),
            "hand": [1.0, 0.5, 2.0, 5.0],
            "hand_at_node": [1.0, 0.5, 2.0, 5.0],
            "river_id": ["tamiang", "tamiang", None, None],
        }
    ).to_parquet(out / "node_hand.parquet")
    pd.DataFrame(
        {
            "node_id": range(4),
            "village_id": ["1114000001", "1114000001", "1114000002", None],
            "population": [10.0, 20.0, 30.0, 0.0],
            "under5": [1.0, 2.0, 3.0, 0.0],
            "elderly": [1.0, 1.0, 1.0, 0.0],
        }
    ).to_parquet(out / "node_population.parquet")
    _gdf(
        {"village_id": ["1114000001", "1114000002"]}, [box(0, -5, 115, 5), box(115, -5, 230, 5)]
    ).to_parquet(out / "villages.parquet")
    _gdf(
        {"hub_id": [1, 2], "is_active": [True, False], "node_id": [0, 3]}, [points[0], points[3]]
    ).to_parquet(out / "logistics_hubs.parquet")
    _gdf(
        {
            "facility_id": [0, 1],
            "type": ["health_center", "school"],
            "node_id": [3, 1],
            "source": ["bpbd", "osm"],
        },
        [points[3], points[1]],
    ).to_parquet(out / "facilities.parquet")
    pd.DataFrame(
        {"facility_id": [0, 1], "hand": [6.0, 0.5], "river_id": [None, "tamiang"]}
    ).to_parquet(out / "facility_hand.parquet")
    return out


def _region(tmp_path):
    path = tmp_path / "aceh-tamiang.yaml"
    path.write_text(yaml.safe_dump(copy.deepcopy(VALID), sort_keys=False), encoding="utf-8")
    return path


def test_assemble_scenario_maps_rivers_villages_hubs_and_health(tmp_path):
    out = _write_inputs(tmp_path / "processed")
    data = builder.assemble_scenario(out, load_region(_region(tmp_path)))
    assert data.graph.hand_min.tolist() == [1.0, 0.0, 4.0]
    assert data.rivers == ["tamiang"]
    assert data.edge_river.tolist() == [0, 0, NO_NODE]
    assert data.node_river.tolist() == [0, 0, NO_NODE, NO_NODE]
    assert data.population.village.tolist() == [0, 0, 1, NO_NODE]
    assert data.hub_nodes == {1: 0}
    assert data.health_nodes.tolist() == [3]
    assert data.health_hand.tolist() == [6.0]
    assert data.health_river.tolist() == [NO_NODE]


@pytest.mark.parametrize(
    ("table", "change", "message"),
    [
        ("edge_hand", lambda t: t.assign(river_id=["krueng", None, None]), "krueng"),
        ("edge_hand", lambda t: t.iloc[::-1].reset_index(drop=True), "order"),
        ("node_population", lambda t: t.assign(village_id=["9", None, None, None]), "village_id"),
    ],
)
def test_assemble_scenario_rejects_inconsistent_tables(tmp_path, table, change, message):
    out = _write_inputs(tmp_path / "processed")
    path = out / f"{table}.parquet"
    change(pd.read_parquet(path)).to_parquet(path)
    with pytest.raises(InvalidInputError, match=message):
        builder.assemble_scenario(out, load_region(_region(tmp_path)))


def test_build_writes_a_readable_bundle(tmp_path):
    out = _write_inputs(tmp_path / "processed")
    path = builder.build(_region(tmp_path), "2026-10-16.1", out, tmp_path / "static")
    loaded = read_bundle(path, "2026-10-16.1")
    assert loaded.manifest.counts["health_facilities"] == 1
    assert set(loaded.manifest.sources) == set(builder.INPUTS)
    np.testing.assert_array_equal(loaded.data.edge_ids, [0, 1, 2])
    assert loaded.sites.facility_ids.tolist() == [0, 1]
    assert loaded.sites.preferred.tolist() == [True, False]
    assert loaded.sites.types == ["health_center", "school"]
    assert loaded.manifest.counts["evacuation_sites"] == 2


def test_build_rejects_bad_versions_and_missing_inputs(tmp_path):
    region = _region(tmp_path)
    with pytest.raises(InvalidInputError, match="invalid data version"):
        builder.build(region, "../x", tmp_path / "processed", tmp_path / "static")
    with pytest.raises(InvalidInputError, match="missing"):
        builder.build(region, "v1", tmp_path / "processed", tmp_path / "static")
