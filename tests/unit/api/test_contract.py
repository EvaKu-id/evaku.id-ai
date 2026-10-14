import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _export_module():
    spec = importlib.util.spec_from_file_location(
        "export_openapi", ROOT / "scripts/export_openapi.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_paths():
    schema = _export_module().create_app().openapi()
    assert set(schema["paths"]) == {"/health", "/simulation", "/route", "/recalc"}


def test_committed_contract_is_up_to_date():
    committed = (ROOT / "contracts/openapi.json").read_text(encoding="utf-8")
    assert committed == _export_module().render(), "run `make openapi` and commit the result"
