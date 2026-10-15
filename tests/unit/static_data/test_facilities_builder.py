import copy
import json

import geopandas as gpd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import shapely
from shapely.geometry import LineString, Point, box

from evaku_ai.core.errors import InvalidInputError
from evaku_ai.domain.enums import FacilityType, HubType
from evaku_ai.domain.region import ManualLocation
from evaku_ai.domain.rules import FacilityRules
from evaku_ai.static_data.builders import facilities as builder
from tests.fixtures.region import VALID

EPSG = 32647
RULES = FacilityRules()


@pytest.mark.parametrize(
    ("tags", "name", "expected"),
    [
        ({"amenity": "hospital"}, "RSUD Muda Sedia", FacilityType.HOSPITAL),
        ({"amenity": "hospital"}, "Puskesmas Banda Mulia", FacilityType.HEALTH_CENTER),
        ({"healthcare": "clinic"}, "", FacilityType.HEALTH_CENTER),
        ({"amenity": "school"}, "SD Negeri 1", FacilityType.SCHOOL),
        ({"building": "mosque"}, "Masjid Raya", FacilityType.PLACE_OF_WORSHIP),
        ({"amenity": "townhall"}, "", FacilityType.VILLAGE_OFFICE),
        ({}, "Kantor Datok Aras Sembilan", FacilityType.VILLAGE_OFFICE),
        ({"leisure": "pitch"}, "", FacilityType.FIELD),
        ({"amenity": "cafe"}, "Kopi", None),
    ],
)
def test_classify_osm(tags, name, expected):
    assert builder.classify_osm(tags, name) == expected


@pytest.mark.parametrize(
    ("category", "name", "expected"),
    [
        ("hospital", "RSUD Aceh Tamiang", FacilityType.HOSPITAL),
        ("hospital", "Klinik Abah", FacilityType.HEALTH_CENTER),
        ("outpatient_care_facility", "", FacilityType.HEALTH_CENTER),
        ("elementary_school", "SDN 2", FacilityType.SCHOOL),
        ("place_of_learning", "Dayah", FacilityType.SCHOOL),
        ("muslim_place_of_worship", "Masjid", FacilityType.PLACE_OF_WORSHIP),
        ("government_office", "Kantor Desa Bundar", FacilityType.VILLAGE_OFFICE),
        ("restaurant", "Warung", None),
    ],
)
def test_classify_overture(category, name, expected):
    assert builder.classify_overture(category, name) == expected


@pytest.mark.parametrize(
    ("name", "tags", "category", "expected"),
    [
        ("Kantor Bupati Aceh Tamiang", {}, "", HubType.REGENT_OFFICE),
        ("Taman Kantor Bupati Aceh Tamiang", {}, "park", None),
        ("BPBD Aceh Tamiang", {}, "", HubType.BPBD_OFFICE),
        ("Kantor Camat Seruway", {}, "", HubType.COMMAND_POST),
        ("", {"aeroway": "aerodrome"}, "", HubType.AIRPORT),
        ("Dermaga", {"amenity": "ferry_terminal"}, "", HubType.PORT),
        ("SD Negeri 1", {"amenity": "school"}, "", None),
    ],
)
def test_classify_hub(name, tags, category, expected):
    assert builder.classify_hub(name, tags, category) == expected


def test_is_government():
    assert builder.is_government("Polsek Rantau", {})
    assert builder.is_government("", {"office": "government"})
    assert builder.is_government("Dinas Pertanahan", {}, "government_office")
    assert not builder.is_government("Masjid", {"amenity": "place_of_worship"})


def poi(source, source_id, x, y, facility=None, hub=None, government=False, name="Tempat"):
    return {
        "source": source,
        "source_id": source_id,
        "name": name,
        "facility_type": facility,
        "hub_type": hub,
        "government": government,
        "geometry": Point(x, y),
    }


def frame(rows):
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def test_dedupe_prefers_source_priority():
    near = 0.0002  # about 22 m
    points = frame(
        [
            poi("overture", "o1", 98.0, 4.3, facility="school"),
            poi("osm", "n1", 98.0 + near, 4.3, facility="school"),
            poi("bpbd", "b1", 98.0, 4.3 + near, facility="school"),
            poi("overture", "o2", 98.01, 4.3, facility="school"),
            poi("overture", "o3", 98.0, 4.3, facility="hospital"),
        ]
    )
    kept = builder.dedupe(points, "facility_type", 50.0, EPSG)
    assert sorted(kept["source_id"]) == ["b1", "o2", "o3"]


def areas():
    villages = gpd.GeoDataFrame(
        {"village_id": ["1114010001", "1114020001"]},
        geometry=[box(98.0, 4.2, 98.1, 4.3), box(98.1, 4.2, 98.2, 4.3)],
        crs="EPSG:4326",
    )
    districts = gpd.GeoDataFrame(
        {"district_id": ["1114010", "1114020"], "name": ["Tamiang Hulu", "Kejuruan Muda"]},
        geometry=[box(98.0, 4.2, 98.1, 4.3), box(98.1, 4.2, 98.2, 4.3)],
        crs="EPSG:4326",
    )
    return villages, districts


def test_assign_areas_and_estimate_hubs():
    villages, districts = areas()
    points = frame(
        [
            poi(
                "overture", "h1", 98.05, 4.25, hub="command_post", name="Kantor Camat Tamiang Hulu"
            ),
            poi("overture", "s1", 98.15, 4.25, facility="school", name="SD 1"),
            poi("osm", "g1", 98.19, 4.29, government=True, name="Polsek Kejuruan Muda"),
            poi("osm", "x1", 98.50, 4.25, facility="school", name="Luar"),
        ]
    )
    located = builder.assign_areas(points, villages, districts)
    assert located.loc[3, "village_id"] is None or np.isnan(located.loc[3, "village_id"])
    assert list(located["district_id"][:3]) == ["1114010", "1114020", "1114020"]
    hubs = located[located["hub_type"].notna()]
    estimated = builder.estimate_hubs(hubs, located, districts, EPSG)
    assert len(estimated) == 2
    added = estimated.iloc[-1]
    assert added["hub_type"] == HubType.COMMAND_POST_ESTIMATED
    assert added["name"] == "Posko perkiraan - Polsek Kejuruan Muda"


def tiny_graph():
    nodes = gpd.GeoDataFrame(
        {"node_id": [0, 1, 2, 3, 4], "x_m": [0.0, 100.0, 200.0, 5000.0, 5100.0]},
        geometry=[
            Point(98.0, 4.25),
            Point(98.001, 4.25),
            Point(98.002, 4.25),
            Point(98.05, 4.25),
            Point(98.051, 4.25),
        ],
        crs="EPSG:4326",
    )
    projected = nodes.to_crs(epsg=EPSG)
    nodes["x_m"], nodes["y_m"] = projected.geometry.x, projected.geometry.y
    edges = gpd.GeoDataFrame(
        {
            "edge_id": [0, 1, 2],
            "u": [0, 1, 3],
            "v": [1, 2, 4],
            "highway_type": ["primary"] * 3,
            "is_bridge": [False] * 3,
            "length_m": [111.0] * 3,
            "travel_s": [8.0] * 3,
            "osm_id": ["1", "1", "2"],
        },
        geometry=[LineString([(0, 0), (1, 1)])] * 3,
        crs="EPSG:4326",
    )
    return nodes, edges


def test_snap_uses_main_component():
    nodes, edges = tiny_graph()
    points = frame([poi("osm", "a", 98.0505, 4.25), poi("osm", "b", 98.0011, 4.2501)])
    snapped = builder.snap_to_graph(points, nodes, edges, EPSG)
    assert list(snapped["node_id"]) == [2, 1]
    assert snapped.loc[0, "snap_m"] > 5000


def test_check_coverage():
    villages, districts = areas()
    facilities = frame(
        [
            poi("osm", "s", 98.05, 4.25, facility="school"),
            poi("osm", "c", 98.15, 4.25, facility="place_of_worship"),
        ]
    )
    facilities = builder.assign_areas(facilities, villages, districts).rename(
        columns={"facility_type": "type"}
    )
    facilities["snap_m"] = [10.0, 3000.0]
    hubs = frame(
        [
            poi("osm", "h", 98.05, 4.25, hub="command_post"),
            poi("osm", "k", 98.15, 4.25, hub="command_post_estimated"),
        ]
    )
    hubs = builder.assign_areas(hubs, villages, districts).rename(columns={"hub_type": "type"})
    hubs["snap_m"] = [5.0, 5.0]
    report = builder.check_coverage(facilities, hubs, districts, RULES)
    assert report.coverage["Tamiang Hulu"] == {"hubs": 1, "evacuation": 1, "health": 0}
    assert report.districts_without_health == ["Kejuruan Muda", "Tamiang Hulu"]
    assert report.far_from_road == 1
    assert report.estimated_hubs == ["Kejuruan Muda: Tempat"]
    with pytest.raises(InvalidInputError, match="Kejuruan Muda"):
        builder.check_coverage(facilities.iloc[:1], hubs, districts, RULES)
    missing = facilities.copy()
    missing.loc[0, "village_id"] = None
    with pytest.raises(InvalidInputError, match="village_id"):
        builder.check_coverage(missing, hubs, districts, RULES)


def test_manual_pois():
    rows = builder.manual_pois(
        [
            ManualLocation(name="Gudang BPBD", lon=98.05, lat=4.25, hub_type=HubType.BPBD_OFFICE),
            ManualLocation(name="SD 9", lon=98.06, lat=4.25, facility_type=FacilityType.SCHOOL),
        ]
    )
    assert list(rows["source"]) == ["bpbd", "bpbd"]
    assert rows.loc[0, "hub_type"] == "bpbd_office"
    assert rows["hub_type"].isna().iloc[1]
    assert builder.manual_pois([]).empty


OSM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="test">
<node id="1" version="1" lat="4.25" lon="98.05">
<tag k="amenity" v="school"/><tag k="name" v="SD 1"/></node>
<node id="2" version="1" lat="4.25" lon="98.06"><tag k="name" v="Kantor Bupati Contoh"/></node>
<node id="10" version="1" lat="4.24" lon="98.14"/><node id="11" version="1" lat="4.24" lon="98.15"/>
<node id="12" version="1" lat="4.25" lon="98.15"/><node id="13" version="1" lat="4.25" lon="98.14"/>
<way id="100" version="1"><nd ref="10"/><nd ref="11"/><nd ref="12"/><nd ref="13"/><nd ref="10"/>
<tag k="building" v="mosque"/><tag k="name" v="Masjid Raya"/></way>
</osm>
"""


def overture_file(path):
    rows = [
        ("o1", "Kantor Camat Kejuruan Muda", "government_office", 0.9, 98.16, 4.26),
        ("o2", "Warung", "restaurant", 0.9, 98.16, 4.27),
        ("o3", "SD Ragu", "school", 0.2, 98.17, 4.27),
    ]
    pq.write_table(
        pa.table(
            {
                "id": [r[0] for r in rows],
                "names": [{"primary": r[1]} for r in rows],
                "basic_category": [r[2] for r in rows],
                "confidence": [r[3] for r in rows],
                "geometry": [shapely.to_wkb(Point(r[4], r[5])) for r in rows],
            }
        ),
        path,
    )
    return path


def test_readers(tmp_path):
    osm = tmp_path / "region.osm"
    osm.write_text(OSM_XML, encoding="utf-8")
    pois = builder.read_osm_pois(osm, (97.9, 4.1, 98.3, 4.4))
    assert set(pois["facility_type"].dropna()) == {"school", "place_of_worship"}
    assert set(pois["hub_type"].dropna()) == {"regent_office"}
    assert "way/100" in set(pois["source_id"])
    overture = builder.read_overture(overture_file(tmp_path / "o.parquet"), RULES)
    assert list(overture["source_id"]) == ["o1", "o2"]
    assert overture.loc[0, "hub_type"] == "command_post"


def test_build_writes_outputs(tmp_path, monkeypatch):
    out = tmp_path / "processed"
    out.mkdir()
    villages, districts = areas()
    villages.to_parquet(out / "villages.parquet")
    districts.to_parquet(out / "districts.parquet")
    nodes, edges = tiny_graph()
    nodes.to_parquet(out / "road_nodes.parquet")
    edges.to_parquet(out / "road_segments.parquet")
    osm = tmp_path / "region.osm"
    osm.write_text(OSM_XML, encoding="utf-8")
    places = overture_file(tmp_path / "o.parquet")
    monkeypatch.setattr(builder, "fetch", lambda remote: osm)
    monkeypatch.setattr(builder, "fetch_places", lambda source, bbox: places)
    config = tmp_path / "aceh-tamiang.yaml"
    config.write_text(json.dumps(copy.deepcopy(VALID), default=str), encoding="utf-8")

    report = builder.build(config, out)
    hubs = gpd.read_parquet(out / "logistics_hubs.parquet")
    facilities = gpd.read_parquet(out / "facilities.parquet")
    assert list(hubs["hub_id"]) == list(range(1, len(hubs) + 1))
    assert set(hubs["type"]) == {"regent_office", "command_post"}
    assert facilities["village_id"].notna().all()
    assert list(facilities["facility_id"]) == list(range(len(facilities)))
    assert report.facilities_by_source == {"osm": 2}
    meta = json.loads((out / "facilities.meta.json").read_text(encoding="utf-8"))
    assert meta["sources"]["overture"]["release"] == "2026-09-23.0"


def test_build_requires_static_layers(tmp_path):
    config = tmp_path / "aceh-tamiang.yaml"
    config.write_text(json.dumps(copy.deepcopy(VALID), default=str), encoding="utf-8")
    with pytest.raises(InvalidInputError, match="road graph first"):
        builder.build(config, tmp_path / "processed")
