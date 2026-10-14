from pathlib import Path

import pytest

from evaku_ai.static_data.builders.tiles import ROADS, VILLAGES, TileLayer, tippecanoe_args


def test_roads_args_keep_only_listed_attributes_and_promote_edge_id():
    args = tippecanoe_args(ROADS, Path("in/roads.geojson"), Path("out"))
    assert args[0] == "tippecanoe"
    assert args[args.index("--output") + 1] == str(Path("out/jalan.pmtiles"))
    assert args[args.index("--layer") + 1] == "ruas"
    assert [a for a in args if a.startswith("--include=")] == [
        "--include=edge_id",
        "--include=highway_type",
        "--include=is_bridge",
    ]
    assert "--use-attribute-for-id=edge_id" in args
    assert args[-1] == str(Path("in/roads.geojson"))


def test_villages_args_have_no_integer_id():
    args = tippecanoe_args(VILLAGES, Path("v.geojson"), Path("out"))
    assert not any(a.startswith("--use-attribute-for-id") for a in args)
    assert args[args.index("--maximum-zoom") + 1] == "13"


def test_args_are_a_list_without_shell_quoting():
    args = tippecanoe_args(VILLAGES, Path("dir with space/v.geojson"), Path("out"))
    assert args[-1] == str(Path("dir with space/v.geojson"))


@pytest.mark.parametrize(("min_zoom", "max_zoom"), [(-1, 10), (12, 10), (5, 23)])
def test_layer_rejects_invalid_zoom(min_zoom, max_zoom):
    with pytest.raises(ValueError, match="zoom"):
        TileLayer("x", "x.pmtiles", ("id",), min_zoom, max_zoom)


def test_layer_rejects_id_outside_attributes():
    with pytest.raises(ValueError, match="attributes"):
        TileLayer("x", "x.pmtiles", ("name",), 0, 10, integer_id="id")
