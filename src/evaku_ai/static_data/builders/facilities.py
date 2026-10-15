from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

from evaku_ai.adapters.download import fetch, file_sha256
from evaku_ai.adapters.overture import fetch_places
from evaku_ai.adapters.region import load_region
from evaku_ai.core.errors import InvalidInputError
from evaku_ai.domain.enums import FacilityType, HubType, LocationSource
from evaku_ai.domain.region import ManualLocation
from evaku_ai.domain.rules import FacilityRules
from evaku_ai.engine.graph import to_csr
from evaku_ai.geo.crs import utm_epsg
from evaku_ai.static_data.builders.graph import parse_tags, region_boundary, to_road_graph

log = logging.getLogger(__name__)

WGS84 = "EPSG:4326"
SOURCE_PRIORITY = (LocationSource.BPBD, LocationSource.OSM, LocationSource.OVERTURE)
EVACUATION_TYPES = {
    FacilityType.SCHOOL,
    FacilityType.PLACE_OF_WORSHIP,
    FacilityType.VILLAGE_OFFICE,
    FacilityType.FIELD,
}
HEALTH_TYPES = {FacilityType.HOSPITAL, FacilityType.HEALTH_CENTER}
OSM_POLYGON_TAGS = ["amenity", "leisure", "building", "office", "aeroway", "landuse"]
HEALTH_NAMES = ("puskesmas", "pustu", "poskesdes", "klinik", "clinic")
VILLAGE_OFFICE_NAMES = ("kantor desa", "kantor datok", "kantor keuchik", "kantor kelurahan")
GOVERNMENT_NAMES = ("kantor", "polsek", "koramil", "polres", "kodim")
GOVERNMENT_CATEGORIES = {"government_office", "police_department", "community_and_government"}
ESTIMATE_ORDER = (
    "government",
    FacilityType.VILLAGE_OFFICE,
    FacilityType.SCHOOL,
    FacilityType.PLACE_OF_WORSHIP,
)
POI_COLUMNS = ["source", "source_id", "name", "facility_type", "hub_type", "government", "geometry"]
FACILITY_COLUMNS = [
    "facility_id",
    "type",
    "name",
    "village_id",
    "district_id",
    "source",
    "source_id",
    "node_id",
    "snap_m",
    "geometry",
]
HUB_COLUMNS = [
    "hub_id",
    "name",
    "type",
    "is_active",
    "source",
    "source_id",
    "district_id",
    "node_id",
    "snap_m",
    "geometry",
]


@dataclass(frozen=True)
class FacilityReport:
    facilities_by_type: dict[str, int]
    facilities_by_source: dict[str, int]
    hubs_by_type: dict[str, int]
    estimated_hubs: list[str]
    coverage: dict[str, dict[str, int]]
    districts_without_health: list[str]
    far_from_road: int


def classify_osm(tags: dict[str, str], name: str) -> FacilityType | None:
    lowered = name.lower()
    amenity, building = tags.get("amenity"), tags.get("building")
    healthcare = tags.get("healthcare")
    # Puskesmas and clinics are often tagged as hospitals; the name is more reliable.
    if (
        amenity in {"clinic", "doctors"}
        or healthcare in {"clinic", "centre", "doctor"}
        or any(n in lowered for n in HEALTH_NAMES)
    ):
        return FacilityType.HEALTH_CENTER
    if amenity == "hospital" or healthcare == "hospital" or building == "hospital":
        return FacilityType.HOSPITAL
    if amenity == "school" or building == "school":
        return FacilityType.SCHOOL
    if amenity == "place_of_worship" or building in {"mosque", "church", "temple"}:
        return FacilityType.PLACE_OF_WORSHIP
    if amenity == "townhall" or any(n in lowered for n in VILLAGE_OFFICE_NAMES):
        return FacilityType.VILLAGE_OFFICE
    if tags.get("leisure") == "pitch":
        return FacilityType.FIELD
    return None


def classify_overture(category: str, name: str) -> FacilityType | None:
    lowered = name.lower()
    if category in {"outpatient_care_facility", "clinic_or_medical_center"} or any(
        n in lowered for n in HEALTH_NAMES
    ):
        return FacilityType.HEALTH_CENTER
    if category == "hospital":
        return FacilityType.HOSPITAL
    if "school" in category or category == "place_of_learning":
        return FacilityType.SCHOOL
    if "place_of_worship" in category:
        return FacilityType.PLACE_OF_WORSHIP
    if any(n in lowered for n in VILLAGE_OFFICE_NAMES):
        return FacilityType.VILLAGE_OFFICE
    return None


def classify_hub(name: str, tags: dict[str, str], category: str = "") -> HubType | None:
    lowered = name.lower()
    if lowered.startswith("kantor bupati"):
        return HubType.REGENT_OFFICE
    if "bpbd" in lowered:
        return HubType.BPBD_OFFICE
    if lowered.startswith("kantor camat"):
        return HubType.COMMAND_POST
    if tags.get("aeroway") == "aerodrome" or category == "airport":
        return HubType.AIRPORT
    if (
        tags.get("amenity") == "ferry_terminal"
        or tags.get("harbour") == "yes"
        or tags.get("landuse") == "port"
    ):
        return HubType.PORT
    return None


def is_government(name: str, tags: dict[str, str], category: str = "") -> bool:
    lowered = name.lower()
    return (
        tags.get("office") == "government"
        or tags.get("amenity") in {"police", "townhall"}
        or category in GOVERNMENT_CATEGORIES
        or any(n in lowered for n in GOVERNMENT_NAMES)
    )


def read_osm_pois(path: Path, bbox: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    frames = []
    for layer, id_columns in (("points", ["osm_id"]), ("multipolygons", ["osm_id", "osm_way_id"])):
        raw = gpd.read_file(path, layer=layer, bbox=bbox)
        rows = []
        for record in raw.to_dict("records"):
            other = record.get("other_tags")
            tags = parse_tags(other if isinstance(other, str) else None)
            tags.update({k: str(record[k]) for k in OSM_POLYGON_TAGS if _present(record.get(k))})
            name = str(record["name"]) if _present(record.get("name")) else ""
            kind = (
                "node"
                if layer == "points"
                else "way"
                if _present(record.get("osm_way_id"))
                else "relation"
            )
            osm_id = next(str(record[c]) for c in reversed(id_columns) if _present(record.get(c)))
            rows.append(
                _poi(LocationSource.OSM, f"{kind}/{osm_id}", name, tags, record["geometry"])
            )
        frames.append(rows)
    return _poi_frame([row for rows in frames for row in rows])


def read_overture(path: Path, rules: FacilityRules) -> gpd.GeoDataFrame:
    table = pq.read_table(path).to_pylist()
    rows = []
    for record in table:
        if (record.get("confidence") or 0) < rules.overture_min_confidence:
            continue
        name = (record.get("names") or {}).get("primary") or ""
        category = record.get("basic_category") or ""
        geometry = shapely.from_wkb(record["geometry"])
        rows.append(_poi(LocationSource.OVERTURE, record["id"], name, {}, geometry, category))
    return _poi_frame(rows)


def manual_pois(locations: list[ManualLocation]) -> gpd.GeoDataFrame:
    rows = [
        {
            "source": LocationSource.BPBD.value,
            "source_id": f"bpbd/{index}",
            "name": loc.name,
            "facility_type": loc.facility_type.value if loc.facility_type else None,
            "hub_type": loc.hub_type.value if loc.hub_type else None,
            "government": loc.hub_type is not None,
            "geometry": shapely.Point(loc.lon, loc.lat),
        }
        for index, loc in enumerate(locations)
    ]
    return _poi_frame(rows)


def dedupe(
    points: gpd.GeoDataFrame, kind_column: str, distance_m: float, epsg: int
) -> gpd.GeoDataFrame:
    if points.empty:
        return points
    rank = points["source"].map({s.value: i for i, s in enumerate(SOURCE_PRIORITY)})
    ordered = points.assign(_rank=rank).sort_values(["_rank", "source_id"]).reset_index(drop=True)
    projected = ordered.to_crs(epsg=epsg)
    xy = np.c_[projected.geometry.x, projected.geometry.y]
    keep = np.zeros(len(ordered), dtype=bool)
    for kind, group in ordered.groupby(kind_column).groups.items():
        kept: list[int] = []
        for index in group:
            if all(np.hypot(*(xy[index] - xy[k])) > distance_m for k in kept):
                kept.append(int(index))
        keep[kept] = True
        log.debug("dedupe %s: %d -> %d", kind, len(group), len(kept))
    return ordered[keep].drop(columns="_rank").reset_index(drop=True)


def assign_areas(
    points: gpd.GeoDataFrame, villages: gpd.GeoDataFrame, districts: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    joined = gpd.sjoin(
        points, villages[["village_id", "geometry"]], how="left", predicate="within"
    ).drop(columns="index_right")
    joined = gpd.sjoin(
        joined, districts[["district_id", "geometry"]], how="left", predicate="within"
    ).drop(columns="index_right")
    return joined[~joined.index.duplicated()].reset_index(drop=True)


def estimate_hubs(
    hubs: gpd.GeoDataFrame, pois: gpd.GeoDataFrame, districts: gpd.GeoDataFrame, epsg: int
) -> gpd.GeoDataFrame:
    covered = set(hubs["district_id"].dropna())
    estimated = []
    for district in districts.itertuples():
        if district.district_id in covered:
            continue
        candidates = pois[pois["district_id"] == district.district_id]
        anchor = gpd.GeoSeries([district.geometry.representative_point()], crs=districts.crs)
        anchor_m = anchor.to_crs(epsg=epsg).iloc[0]
        for kind in ESTIMATE_ORDER:
            pool = (
                candidates[candidates["government"]]
                if kind == "government"
                else candidates[candidates["facility_type"] == kind]
            )
            pool = pool[pool["name"].str.len() > 0]
            if pool.empty:
                continue
            best = pool.iloc[int(np.argmin(pool.to_crs(epsg=epsg).distance(anchor_m).to_numpy()))]
            estimated.append(
                {
                    **best.to_dict(),
                    "name": f"Posko perkiraan - {best['name']}",
                    "hub_type": HubType.COMMAND_POST_ESTIMATED.value,
                }
            )
            break
    if not estimated:
        return hubs
    extra = gpd.GeoDataFrame(estimated, geometry="geometry", crs=hubs.crs)
    return pd.concat([hubs, extra], ignore_index=True)


def snap_to_graph(
    points: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame, epsg: int
) -> gpd.GeoDataFrame:
    graph = to_road_graph(nodes, edges)
    _, labels = csgraph.connected_components(to_csr(graph), directed=False)
    main = np.flatnonzero(labels == np.bincount(labels).argmax())
    tree = cKDTree(np.c_[graph.node_x[main], graph.node_y[main]])
    projected = points.to_crs(epsg=epsg)
    distance, index = tree.query(np.c_[projected.geometry.x, projected.geometry.y])
    return points.assign(node_id=main[index].astype(np.int64), snap_m=np.round(distance, 1))


def check_coverage(
    facilities: gpd.GeoDataFrame,
    hubs: gpd.GeoDataFrame,
    districts: gpd.GeoDataFrame,
    rules: FacilityRules,
) -> FacilityReport:
    if facilities["village_id"].isna().any():
        raise InvalidInputError("every facility needs a village_id")
    coverage: dict[str, dict[str, int]] = {}
    for district in districts.itertuples():
        in_district = facilities[facilities["district_id"] == district.district_id]
        coverage[district.name] = {
            "hubs": int((hubs["district_id"] == district.district_id).sum()),
            "evacuation": int(in_district["type"].isin(EVACUATION_TYPES).sum()),
            "health": int(in_district["type"].isin(HEALTH_TYPES).sum()),
        }
    missing = [name for name, c in coverage.items() if c["hubs"] == 0 or c["evacuation"] == 0]
    if missing:
        raise InvalidInputError(f"districts without a logistics hub or evacuation site: {missing}")
    estimated = hubs[hubs["type"] == HubType.COMMAND_POST_ESTIMATED.value]
    names = dict(zip(districts["district_id"], districts["name"], strict=True))
    far = int(
        (facilities["snap_m"] > rules.max_snap_m).sum() + (hubs["snap_m"] > rules.max_snap_m).sum()
    )
    return FacilityReport(
        facilities_by_type=dict(Counter(facilities["type"])),
        facilities_by_source=dict(Counter(facilities["source"])),
        hubs_by_type=dict(Counter(hubs["type"])),
        estimated_hubs=[
            f"{names[d]}: {n}"
            for d, n in zip(estimated["district_id"], estimated["name"], strict=True)
        ],
        coverage=coverage,
        districts_without_health=sorted(n for n, c in coverage.items() if c["health"] == 0),
        far_from_road=far,
    )


def build(region_path: Path, out_dir: Path = Path("data/processed")) -> FacilityReport:
    region = load_region(region_path)
    rules = region.facility_rules()
    epsg = utm_epsg(*region.center)
    villages, districts, nodes, edges = _static_layers(out_dir)
    boundary = region_boundary(districts, region.road_network().buffer_m, epsg)
    min_x, min_y, max_x, max_y = (float(b) for b in boundary.bounds)
    bbox = (min_x, min_y, max_x, max_y)

    overture_path = fetch_places(region.inputs.overture_places, bbox)
    pois = pd.concat(
        [
            manual_pois(region.bpbd_locations),
            read_osm_pois(fetch(region.inputs.osm_extract), bbox),
            read_overture(overture_path, rules),
        ],
        ignore_index=True,
    )
    pois["government"] = pois["government"].fillna(False).astype(bool)
    pois = assign_areas(gpd.GeoDataFrame(pois, geometry="geometry", crs=WGS84), villages, districts)
    pois = pois[pois.within(boundary)].reset_index(drop=True)

    facilities = dedupe(
        pois[pois["facility_type"].notna() & pois["village_id"].notna()],
        "facility_type",
        rules.dedupe_m,
        epsg,
    )
    # Hubs outside the regency (neighbouring district offices, ports) are not its command posts.
    in_regency = pois["district_id"].notna()
    hubs = dedupe(pois[pois["hub_type"].notna() & in_regency], "hub_type", rules.dedupe_m, epsg)
    hubs = estimate_hubs(hubs, pois, districts, epsg)
    facilities = _finish_facilities(snap_to_graph(facilities, nodes, edges, epsg))
    hubs = _finish_hubs(snap_to_graph(hubs, nodes, edges, epsg))
    report = check_coverage(facilities, hubs, districts, rules)

    facilities.to_parquet(out_dir / "facilities.parquet")
    hubs.to_parquet(out_dir / "logistics_hubs.parquet")
    meta = {
        "region_id": region.region_id,
        "sources": {
            "osm": {
                "license": region.inputs.osm_extract.license,
                "sha256": region.inputs.osm_extract.sha256,
            },
            "overture": {
                "release": region.inputs.overture_places.release,
                "license": region.inputs.overture_places.license,
                "sha256": file_sha256(overture_path),
            },
            "bpbd": {"locations": len(region.bpbd_locations)},
        },
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **asdict(report),
    }
    (out_dir / "facilities.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info(
        "facilities %d, hubs %d (%d estimated)",
        len(facilities),
        len(hubs),
        len(report.estimated_hubs),
    )
    return report


def _static_layers(
    out_dir: Path,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    names = ("villages", "districts", "road_nodes", "road_segments")
    paths = [out_dir / f"{name}.parquet" for name in names]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise InvalidInputError(f"missing {missing}; build villages and the road graph first")
    villages, districts, nodes, edges = (gpd.read_parquet(p) for p in paths)
    return villages, districts, nodes, edges


def _finish_facilities(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = points.rename(columns={"facility_type": "type"}).sort_values(
        ["type", "name", "source_id"]
    )
    out = out.reset_index(drop=True)
    out["facility_id"] = np.arange(len(out), dtype=np.int64)
    return out[FACILITY_COLUMNS]


def _finish_hubs(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = points.rename(columns={"hub_type": "type"}).sort_values(["type", "name", "source_id"])
    out = out.reset_index(drop=True)
    out["hub_id"] = np.arange(1, len(out) + 1, dtype=np.int64)
    out["is_active"] = True
    return out[HUB_COLUMNS]


def _poi(
    source: LocationSource,
    source_id: str,
    name: str,
    tags: dict[str, str],
    geometry: shapely.Geometry,
    category: str = "",
) -> dict[str, object]:
    facility = (
        classify_osm(tags, name)
        if source is LocationSource.OSM
        else classify_overture(category, name)
    )
    hub = classify_hub(name, tags, category)
    return {
        "source": source.value,
        "source_id": source_id,
        "name": name.strip(),
        "facility_type": facility.value if facility else None,
        "hub_type": hub.value if hub else None,
        "government": is_government(name, tags, category),
        "geometry": geometry.representative_point(),
    }


def _poi_frame(rows: list[dict[str, object]]) -> gpd.GeoDataFrame:
    if not rows:
        return gpd.GeoDataFrame({c: [] for c in POI_COLUMNS[:-1]}, geometry=[], crs=WGS84)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=WGS84)[POI_COLUMNS]


def _present(value: object) -> bool:
    return value is not None and not (isinstance(value, float) and np.isnan(value)) and value != ""
