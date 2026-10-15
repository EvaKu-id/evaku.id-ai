from pathlib import Path

import typer

from evaku_ai.static_data import release
from evaku_ai.static_data.builders import (
    bundle,
    facilities,
    graph,
    hand,
    hand_objects,
    population,
    tiles,
    villages,
)

app = typer.Typer(help="Build and release static data", no_args_is_help=True)

REGION = typer.Option(Path("config/regions/aceh-tamiang.yaml"), help="Region config file")


@app.command("build-villages", help="Village and district boundaries from the configured source")
def build_villages(region: Path = REGION) -> None:
    villages.build(region)


@app.command("build-graph", help="Road graph from the OSM .pbf extract")
def build_graph(region: Path = REGION) -> None:
    graph.build(region)


@app.command("build-facilities", help="Facilities and logistics hubs from OSM")
def build_facilities(region: Path = REGION) -> None:
    facilities.build(region)


@app.command("build-population", help="WorldPop clipped and snapped to road nodes")
def build_population(region: Path = REGION) -> None:
    population.build(region)


@app.command("build-hand", help="HAND raster, streams, and river basins from the DEM")
def build_hand(region: Path = REGION) -> None:
    hand.build(region)


@app.command("build-hand-objects", help="HAND per road edge, node, facility, and village")
def build_hand_objects(region: Path = REGION) -> None:
    hand_objects.build(region)


@app.command("build-bundle", help="Engine bundle data/static/{version}/ for the model API")
def build_bundle(version: str, region: Path = REGION) -> None:
    bundle.build(region, version)


@app.command("build-tiles", help="Vector tiles jalan.pmtiles and desa.pmtiles for the web map")
def build_tiles(region: Path = REGION) -> None:
    tiles.build(region)


@app.command("release", help="Upload data/static/{version}/ to R2")
def release_cmd(version: str, region: Path = REGION) -> None:
    release.run(version, region)


if __name__ == "__main__":
    app()
