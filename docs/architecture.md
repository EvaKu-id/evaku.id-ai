# Architecture

## Place in the system

EvaKu has three repos:

| Repo | Owner | Contents |
| --- | --- | --- |
| `evaku-web` | FE, BE | Next.js UI and main backend, Supabase migrations, `evaku.dbml` |
| `evaku-ai` | AI/ML | this repo: access engine, model API, pipeline, static data |
| `evaku-ml` | ML | `evaku_ml` package (features, radar preprocessing), notebooks, model release |

The Next.js backend on Vercel is the main backend and the only caller of the model API. Browsers
never call the model API, and normal pages never depend on it: only free simulation and on-demand
routes do, and both fall back to precomputed results in R2 when the model API is down.

## Role of this repo

`evaku-ai` owns everything Python that runs in production for EvaKu:

- **Model API** (FastAPI on Cloud Run): interactive simulation and routing, called only by the
  Next.js backend with the `X-Model-Key` header.
- **Pipeline** (GitHub Actions, every 6 hours): forecast and radar runs written to Supabase and R2.
- **Static data** (one-off): road graph, facilities, population, HAND, released to R2.

Model training is NOT here. The `evaku-ml` repo releases model packages to R2 `model/{version}/`
and the shared `evaku_ml` package (features and preprocessing), so features are identical in
training and in the pipeline. This repo pins both.

## Layers

| Layer | Package | May import | Notes |
| --- | --- | --- | --- |
| Entrypoints | `api`, `pipeline`, `static_data` | everything below | wiring only, little logic |
| I/O and models | `adapters`, `inference` | `engine`, `geo`, `domain`, `core` | databases, R2, GEE, Open-Meteo, ONNX, LightGBM |
| Pure logic | `engine`, `geo` | `domain`, `core` | numpy/scipy, no I/O, unit-tested on small graphs |
| Vocabulary | `domain` | `core` | enums, fixed rules, result types |
| Foundation | `core` | nothing in the package | settings, logging, errors |

`lint-imports` enforces this in CI (contracts in `pyproject.toml`).

## Data flow

0. Every step reads the active region from `REGION_CONFIG` (default
   `config/regions/aceh-tamiang.yaml`) through `adapters/region.load_region`, which parses it with
   `yaml.safe_load` and validates it as `domain/region.RegionConfig` (bbox, DEM extent, rivers,
   replay dates, optional rule overrides; rule defaults stay in `domain/rules.py`).
   Village and district boundaries come from the OCHA HDX COD-AB dataset (BPS, CC BY-IGO;
   attribution required); `village_id` is the 10-digit BPS code.
   The road graph is cut from a pinned Geofabrik OpenStreetMap snapshot (ODbL; maps must credit
   "© OpenStreetMap contributors") and stored as `road_nodes`/`road_segments` GeoParquet.
   Facilities and logistics hubs combine OSM with Overture Maps Places (pinned release; CDLA
   Permissive 2.0 / Apache 2.0 / CC0 by source), with official BPBD locations from the region config
   taking priority.
   Population comes from WorldPop Global2 R2025A constrained rasters (DOI 10.5258/SOTON/WP00841,
   CC BY 4.0; attribution required): the 100 m total is split into under-5 and 60+ counts with
   age shares from the 1 km age bands, then each pixel is assigned to the nearest road node in
   the same village.
   HAND is computed at 30 m in UTM from Copernicus DEM GLO-30 tiles on AWS Open Data (pinned
   checksums; attribution required: "© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH
   2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved") with
   WhiteboxTools 2.4.0, run as a checksum-pinned conda-forge binary. OSM rivers are burned into the
   DEM to steer flow (HAND itself is measured on the unburned surface) and always count as streams.
   The same run delineates the
   basin of each configured river from its GloFAS point down to the sea; objects outside a basin
   are not flooded by that river in simulations.
   HAND is then attached to objects: the minimum along each road edge (sampled every 15 m, skipping
   45 m around road-stream crossings that are culverts or untagged bridges), the population-weighted
   median of the WorldPop pixels behind each road node, the value at each facility, and a median per
   village. Each object also records the river basin it belongs to.
   Radar runs reach the same road status through a depth raster (meters, EPSG:32647, nodata outside
   the image): the maximum depth along each edge goes through the same bridge and threshold rules
   as simulations.
   Forecast runs also plan evacuation: for each high-probability village, up to three facilities
   that stay above the forecast water level plus a safety margin, reached by road under current
   conditions, with the first horizon at which the shown route is cut.
1. `static_data` builds the graph, HAND, population, and facilities, then releases them to R2
   `data/static/{version}/`.
2. The `api` image loads that bundle into memory at startup. Its engine part is `scenario.npz`
   (plain arrays, no geometry, read without pickle and without pyarrow) plus `manifest.json`
   (version, SHA-256, counts, source checksums). In production a missing or mismatched bundle stops
   startup. `/simulation` turns a water level
   into road status, reachability, and isolated villages; `/route` returns a land or land + boat
   route.
3. `pipeline` runs forecast (Open-Meteo + GEE features -> LightGBM -> water level -> engine) and
   radar (Sentinel-1 -> U-Net -> depth -> engine), writes one run per radar scene or forecast
   horizon as `draft`, flips it to `published` in the same transaction, uploads flood PMTiles,
   then calls the web app's `/api/pipeline-selesai`.

Forecast, radar, and simulation share one access engine; they differ only in where the flood
depth comes from.

## Contracts and artifacts

- API: `contracts/openapi.json`, generated from `api/schemas.py`. Field names are English and
  match `evaku.dbml`. `evaku-web` generates its TypeScript types from this file.
- Database: schema owned by `evaku-web` (`supabase/migrations`, `evaku.dbml`); this repo only
  writes rows.
- Features: `contracts/fitur.json` in `evaku-ml`, computed by the `evaku_ml` package pinned by tag.
- Model packages: R2 `model/{version}/`, selected with `MODEL_VERSION`, verified against
  `manifest.json` checksums.
- Static data: R2 `data/static/{version}/`: GeoParquet for the pipeline and ML, and
  `scenario.npz` + `manifest.json` for the `api` image.
- Other R2 paths: `peta/*.pmtiles` (maps) and `cadangan/...` (fallback scenarios, replay copies).
