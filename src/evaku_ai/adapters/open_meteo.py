import pandas as pd


def precipitation_forecast(points: list[tuple[float, float]], hours: int = 72) -> pd.DataFrame:
    raise NotImplementedError


def river_discharge(
    lon: float, lat: float, past_days: int = 14, forecast_days: int = 3
) -> pd.DataFrame:
    raise NotImplementedError
