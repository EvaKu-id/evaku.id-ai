import pytest

from evaku_ai.geo.crs import utm_epsg


@pytest.mark.parametrize(
    ("lon", "lat", "epsg"),
    [(115.2, -2.7, 32750), (106.8, -6.2, 32748), (98.7, 3.6, 32647)],
)
def test_utm_epsg(lon, lat, epsg):
    assert utm_epsg(lon, lat) == epsg


@pytest.mark.parametrize(("lon", "lat"), [(181.0, 0.0), (0.0, -91.0)])
def test_utm_epsg_rejects_invalid_coordinates(lon, lat):
    with pytest.raises(ValueError):
        utm_epsg(lon, lat)
