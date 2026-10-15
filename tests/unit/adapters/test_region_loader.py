from datetime import date
from pathlib import Path

import pytest

from evaku_ai.adapters.region import MAX_CONFIG_BYTES, load_region
from evaku_ai.core.config import Settings
from evaku_ai.core.errors import ConfigurationError
from evaku_ai.geo.crs import utm_epsg

REGIONS = Path("config/regions")
REPLAY_SITES = [(98.06, 4.28), (97.98, 4.20), (98.17, 4.40)]


def test_aceh_tamiang_config_loads():
    region = load_region(REGIONS / "aceh-tamiang.yaml")
    assert (region.region_id, region.regency_code) == ("aceh-tamiang", "11.16")
    assert utm_epsg(*region.center) == 32647
    min_lon, min_lat, max_lon, max_lat = region.bbox
    for lon, lat in REPLAY_SITES:
        assert min_lon <= lon <= max_lon and min_lat <= lat <= max_lat
    assert region.replay.peak_date == date(2026, 10, 13)
    assert region.replay.start_date == date(2026, 10, 10)
    assert region.replay_backup is not None
    assert region.replay_backup.peak_date == date(2025, 11, 28)


def test_example_template_stays_valid():
    assert load_region(REGIONS / "example.yaml").region_id == "example"


def test_default_setting_points_to_existing_region():
    assert Settings(_env_file=None).region_config.is_file()


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_missing_file(tmp_path):
    with pytest.raises(ConfigurationError, match="not found"):
        load_region(tmp_path / "nowhere.yaml")


def test_oversized_file(tmp_path):
    path = write(tmp_path, "big.yaml", "#" * (MAX_CONFIG_BYTES + 1))
    with pytest.raises(ConfigurationError, match="larger than"):
        load_region(path)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("region_id: [unclosed", "not valid YAML"),
        ("!!python/object/apply:os.system ['echo hacked']", "not valid YAML"),
        ("- just\n- a list\n", "must be a mapping"),
        ("region_id: aceh-tamiang\nbbox: [1, 2]\n", "invalid region config"),
    ],
)
def test_bad_content(tmp_path, text, message):
    with pytest.raises(ConfigurationError, match=message):
        load_region(write(tmp_path, "aceh-tamiang.yaml", text))


def test_validation_message_names_file_and_field(tmp_path):
    path = write(tmp_path, "aceh-tamiang.yaml", "region_id: aceh-tamiang\n")
    with pytest.raises(ConfigurationError) as info:
        load_region(path)
    assert str(path) in str(info.value)
    assert "bbox: Field required" in str(info.value)


def test_region_id_must_match_file_name(tmp_path):
    text = (REGIONS / "aceh-tamiang.yaml").read_text(encoding="utf-8")
    path = write(tmp_path, "other.yaml", text)
    with pytest.raises(ConfigurationError, match="does not match file name"):
        load_region(path)
