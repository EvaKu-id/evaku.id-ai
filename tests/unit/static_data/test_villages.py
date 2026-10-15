import copy
import json

import geopandas as gpd
import pytest
from shapely.geometry import MultiPolygon, Polygon, box

from evaku_ai.core.errors import InvalidInputError
from evaku_ai.domain.region import RegionConfig
from evaku_ai.static_data.builders import villages as builder
from tests.fixtures.region import VALID

REGION_DATA = copy.deepcopy(VALID)
KUALA_SIMPANG = box(98.05, 4.27, 98.07, 4.29)
BOWTIE = Polygon([(98.00, 4.30), (98.02, 4.32), (98.02, 4.30), (98.00, 4.32)])


def region():
    return RegionConfig.model_validate(REGION_DATA)


def admin4():
    rows = [
        ("ID1114040001", "Sriwijaya", "ID1114040", "Kota Kuala Simpang", KUALA_SIMPANG),
        ("ID1114040002", "Kota Lintang", "ID1114040", "Kota Kuala Simpang", BOWTIE),
        ("ID1114010002", " Rongoh ", "ID1114010", "Tamiang Hulu", box(97.92, 4.10, 97.95, 4.13)),
        ("ID1205010001", "Sekoci", "ID1205010", "Besitang", box(98.03, 4.02, 98.05, 4.04)),
    ]
    return gpd.GeoDataFrame(
        {
            "adm4_pcode": [r[0] for r in rows],
            "adm4_name": [r[1] for r in rows],
            "adm3_pcode": [r[2] for r in rows],
            "adm3_name": [r[3] for r in rows],
            "adm2_pcode": ["ID1114", "ID1114", "ID1114", "ID1205"],
            "adm2_name": ["Aceh Tamiang"] * 3 + ["Langkat"],
            "area_sqkm": [1.0] * 4,
        },
        geometry=[r[4] for r in rows],
        crs="EPSG:4326",
    )


def admin3():
    return gpd.GeoDataFrame(
        {
            "adm3_pcode": ["ID1114040", "ID1114010", "ID1205010"],
            "adm3_name": ["Kota Kuala Simpang", "Tamiang Hulu", "Besitang"],
            "adm2_pcode": ["ID1114", "ID1114", "ID1205"],
            "adm2_name": ["Aceh Tamiang", "Aceh Tamiang", "Langkat"],
        },
        geometry=[
            box(98.04, 4.26, 98.08, 4.33),
            box(97.90, 4.08, 97.96, 4.15),
            box(98.0, 4.0, 98.1, 4.1),
        ],
        crs="EPSG:4326",
    )


@pytest.fixture
def normalized():
    villages = builder.normalize_villages(builder.select_regency(admin4(), "1114"))
    districts = builder.normalize_districts(builder.select_regency(admin3(), "1114"))
    return villages, districts


def test_select_regency_drops_other_regencies():
    assert set(builder.select_regency(admin4(), "1114")["adm2_pcode"]) == {"ID1114"}


def test_normalize_villages(normalized):
    villages, _ = normalized
    assert list(villages.columns) == builder.VILLAGE_COLUMNS
    assert list(villages["village_id"]) == ["1114040001", "1114040002", "1114010002"]
    assert villages.loc[2, "name"] == "Rongoh"
    assert list(villages["district_id"]) == ["1114040", "1114040", "1114010"]
    assert (villages.geom_type == "MultiPolygon").all()
    assert villages.is_valid.all()
    assert villages.crs.to_epsg() == 4326


def test_projected_input_is_reprojected():
    projected = admin4().to_crs(32647)
    villages = builder.normalize_villages(builder.select_regency(projected, "1114"))
    assert villages.crs.to_epsg() == 4326
    assert villages.total_bounds[0] == pytest.approx(97.92, abs=1e-6)


def test_check_villages_reports(normalized):
    villages, districts = normalized
    report = builder.check_villages(villages, districts, region())
    assert (report.villages, report.districts) == (3, 2)
    assert report.area_km2 > 0
    assert report.duplicate_names == []
    assert report.bounds[0] == pytest.approx(97.92)


def _broken(normalized, column, index, value):
    villages, districts = normalized
    villages = villages.copy()
    villages.loc[index, column] = value
    return villages, districts


@pytest.mark.parametrize(
    ("column", "index", "value", "message"),
    [
        ("village_id", 1, "1114040001", "duplicate village_id"),
        ("village_id", 0, "111404001", "6 digits"),
        ("village_id", 0, "1205040001", "6 digits"),
        ("district_id", 0, "111404", "7 digits"),
        ("district_id", 0, "1114010", "start with its district_id"),
        ("name", 0, "", "blank name"),
    ],
)
def test_check_villages_rejects_bad_codes(normalized, column, index, value, message):
    villages, districts = _broken(normalized, column, index, value)
    with pytest.raises(InvalidInputError, match=message):
        builder.check_villages(villages, districts, region())


def test_check_villages_rejects_unknown_district(normalized):
    villages, districts = normalized
    with pytest.raises(InvalidInputError, match="unknown districts"):
        builder.check_villages(villages, districts.iloc[:1], region())


def test_check_villages_rejects_outside_bbox(normalized):
    villages, districts = normalized
    villages = villages.copy()
    villages.loc[0, "geometry"] = MultiPolygon([box(99.0, 4.0, 99.1, 4.1)])
    with pytest.raises(InvalidInputError, match="outside region bbox"):
        builder.check_villages(villages, districts, region())


def test_check_villages_rejects_empty(normalized):
    villages, districts = normalized
    with pytest.raises(InvalidInputError, match="no villages"):
        builder.check_villages(villages.iloc[:0], districts, region())


def test_as_multipolygon_keeps_polygons_only():
    from shapely.geometry import GeometryCollection, LineString

    mixed = GeometryCollection([box(0, 0, 1, 1), LineString([(0, 0), (2, 2)])])
    assert builder._as_multipolygon(mixed).geom_type == "MultiPolygon"
    with pytest.raises(InvalidInputError, match="no polygon"):
        builder._as_multipolygon(LineString([(0, 0), (1, 1)]))


def write_gpkg(path):
    admin4().to_file(path, layer="idn_admin4", driver="GPKG")
    admin3().to_file(path, layer="idn_admin3", driver="GPKG")
    outline = [box(0, 0, 1, 1).boundary]
    lines = gpd.GeoDataFrame({"name": ["line"]}, geometry=outline, crs="EPSG:4326")
    lines.to_file(path, layer="idn_adminlines", driver="GPKG")


def test_read_admin_level_picks_layer_and_filters_bbox(tmp_path):
    path = tmp_path / "admin.gpkg"
    write_gpkg(path)
    level4 = builder.read_admin_level(path, 4, (97.70, 4.05, 98.31, 4.56))
    assert set(level4["adm4_pcode"]) == {"ID1114040001", "ID1114040002", "ID1114010002"}
    assert "adm3_pcode" in builder.read_admin_level(path, 3, (97.0, 3.0, 99.0, 5.0)).columns
    with pytest.raises(InvalidInputError, match="no admin level 5"):
        builder.read_admin_level(path, 5, (97.0, 3.0, 99.0, 5.0))


def test_build_writes_outputs(tmp_path, monkeypatch):
    gpkg = tmp_path / "admin.gpkg"
    write_gpkg(gpkg)
    config = tmp_path / "aceh-tamiang.yaml"
    config.write_text(json.dumps(REGION_DATA, default=str), encoding="utf-8")
    monkeypatch.setattr(builder, "fetch", lambda remote: gpkg)
    out = tmp_path / "processed"
    report = builder.build(config, out)
    assert (report.villages, report.districts) == (3, 2)
    saved = gpd.read_parquet(out / "villages.parquet")
    assert list(saved.columns) == builder.VILLAGE_COLUMNS
    assert len(gpd.read_parquet(out / "districts.parquet")) == 2
    meta = json.loads((out / "villages.meta.json").read_text(encoding="utf-8"))
    assert meta["license"] == "CC BY-IGO"
    assert meta["villages"] == 3
