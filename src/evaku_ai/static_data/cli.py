from pathlib import Path

import typer

from evaku_ai.static_data import release
from evaku_ai.static_data.builders import facilities, graph, hand, population, tiles

app = typer.Typer(help="Build and release static data", no_args_is_help=True)

REGION = typer.Option(Path("config/regions/example.yaml"), help="Region config file")


@app.command("build-graph", help="Road graph from the OSM .pbf extract")
def build_graph(region: Path = REGION) -> None:
    graph.build(region)


@app.command("build-facilities", help="Facilities and logistics hubs from OSM")
def build_facilities(region: Path = REGION) -> None:
    facilities.build(region)


@app.command("build-population", help="WorldPop clipped and snapped to road nodes")
def build_population(region: Path = REGION) -> None:
    population.build(region)


@app.command("build-hand", help="HAND raster and HAND per edge, population pixel, facility")
def build_hand(region: Path = REGION) -> None:
    hand.build(region)


@app.command("build-tiles", help="Vector tiles jalan.pmtiles and desa.pmtiles for the web map")
def build_tiles(region: Path = REGION) -> None:
    tiles.build(region)


@app.command("release", help="Upload data/static/{version}/ to R2")
def release_cmd(version: str, region: Path = REGION) -> None:
    release.run(version, region)


if __name__ == "__main__":
    app()
