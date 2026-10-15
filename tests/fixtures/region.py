from datetime import date


def _remote(path, digit):
    return {
        "path": path,
        "url": f"https://example.org/{path.rsplit('/', 1)[-1]}",
        "sha256": digit * 64,
        "source": "WorldPop",
        "license": "CC BY 4.0",
    }


VALID = {
    "region_id": "aceh-tamiang",
    "name": "Kabupaten Aceh Tamiang",
    "province": "Aceh",
    "regency_code": "11.16",
    "bps_code": "1114",
    "bbox": [97.70, 3.86, 98.31, 4.56],
    "dem_bbox": [97.20, 3.60, 98.40, 4.65],
    "rivers": [{"river_id": "tamiang", "name": "Sungai Tamiang", "glofas_point": [98.125, 4.375]}],
    "replay": {"name": "Banjir Oktober 2026", "peak_date": date(2026, 10, 13), "days_before": 3},
    "replay_backup": {"name": "Banjir November 2025", "peak_date": date(2025, 11, 28)},
    "inputs": {
        "osm_extract": {
            "path": "data/raw/region.osm.pbf",
            "url": "https://example.org/region.osm.pbf",
            "sha256": "b" * 64,
            "source": "OpenStreetMap contributors",
            "license": "ODbL 1.0",
        },
        "population": {
            "year": 2026,
            "total_100m": _remote("data/raw/pop_100m.tif", "c"),
            "total_1km": _remote("data/raw/pop_1km.tif", "d"),
            "under5_1km": [_remote("data/raw/t_00.tif", "e")],
            "elderly_1km": [_remote("data/raw/t_60.tif", "f")],
        },
        "dem": [_remote("data/raw/dem_n03_e097.tif", "1")],
        "overture_places": {
            "release": "2026-09-23.0",
            "path": "data/raw/overture.parquet",
            "license": "CDLA Permissive 2.0",
        },
        "village_boundaries": {
            "path": "data/raw/admin.gdb.zip",
            "url": "https://example.org/admin.gdb.zip",
            "sha256": "a" * 64,
            "source": "BPS via OCHA HDX",
            "license": "CC BY-IGO",
        },
    },
}
