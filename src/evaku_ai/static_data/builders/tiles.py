from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TileLayer:
    name: str
    filename: str
    attributes: tuple[str, ...]
    min_zoom: int
    max_zoom: int
    integer_id: str | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.min_zoom <= self.max_zoom <= 22:
            raise ValueError(f"invalid zoom range {self.min_zoom}..{self.max_zoom}")
        if self.integer_id is not None and self.integer_id not in self.attributes:
            raise ValueError(f"id attribute {self.integer_id!r} must be in attributes")


ROADS = TileLayer(
    name="ruas",
    filename="jalan.pmtiles",
    attributes=("edge_id", "highway_type", "is_bridge"),
    min_zoom=8,
    max_zoom=14,
    integer_id="edge_id",
)
VILLAGES = TileLayer(
    name="desa",
    filename="desa.pmtiles",
    attributes=("village_id", "name"),
    min_zoom=7,
    max_zoom=13,
)


def tippecanoe_args(layer: TileLayer, source: Path, out_dir: Path) -> list[str]:
    args = [
        "tippecanoe",
        "--output",
        str(out_dir / layer.filename),
        "--force",
        "--layer",
        layer.name,
        "--minimum-zoom",
        str(layer.min_zoom),
        "--maximum-zoom",
        str(layer.max_zoom),
        "--drop-densest-as-needed",
        *(f"--include={attribute}" for attribute in layer.attributes),
    ]
    if layer.integer_id is not None:
        args.append(f"--use-attribute-for-id={layer.integer_id}")
    args.append(str(source))
    return args


def build(region: Path) -> None:
    raise NotImplementedError
