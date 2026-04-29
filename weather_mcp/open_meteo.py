"""Open-Meteo API client (no API key required).

Endpoints used:
- geocoding-api.open-meteo.com        → city name → lat/lon
- archive-api.open-meteo.com          → historical daily weather (training data)
- api.open-meteo.com/v1/forecast      → 7-day forecast (inference data)
- air-quality-api.open-meteo.com      → PM2.5, PM10, AQI (forecast + history)
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import httpx
import pandas as pd

RAIN_CODES: set[int] = {
    51, 53, 55, 56, 57,
    61, 63, 65, 66, 67,
    71, 73, 75, 77,
    80, 81, 82,
    85, 86,
    95, 96, 99,
}

FEATURES: list[str] = ["precipitation", "temp_max", "temp_min", "windspeed"]

GEOCODE_URL     = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL     = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL    = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

_TIMEOUT = httpx.Timeout(20.0)


async def geocode_city(city: str) -> dict[str, Any]:
    """Resolve a city name to coordinates and a canonical label."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.get(GEOCODE_URL, params={"name": city, "count": 1})
        r.raise_for_status()
        data = r.json()
    results = data.get("results") or []
    if not results:
        raise ValueError(f"City not found: {city!r}")
    top = results[0]
    label = ", ".join(p for p in [top.get("name"), top.get("admin1"), top.get("country")] if p)
    return {
        "latitude": float(top["latitude"]),
        "longitude": float(top["longitude"]),
        "label": label,
        "timezone": top.get("timezone", "UTC"),
    }


async def fetch_historical_weather(
    latitude: float,
    longitude: float,
    years: int = 2,
) -> pd.DataFrame:
    """Daily historical weather including UV index and apparent temperature.

    Columns: date, precipitation, temp_max, temp_min, windspeed,
             uv_index, apparent_temp_max, weather_code, rained
    """
    end = dt.date.today() - dt.timedelta(days=2)
    start = end - dt.timedelta(days=365 * years)
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join([
            "weather_code",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
            "uv_index_max",
            "apparent_temperature_max",
        ]),
        "timezone": "auto",
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.get(ARCHIVE_URL, params=params)
        r.raise_for_status()
        data = r.json()

    daily = data["daily"]
    n = len(daily["time"])
    df = pd.DataFrame({
        "date":             pd.to_datetime(daily["time"]),
        "precipitation":    daily["precipitation_sum"],
        "temp_max":         daily["temperature_2m_max"],
        "temp_min":         daily["temperature_2m_min"],
        "windspeed":        daily["wind_speed_10m_max"],
        "uv_index":         daily.get("uv_index_max") or [None] * n,
        "apparent_temp_max":daily.get("apparent_temperature_max") or [None] * n,
        "weather_code":     daily["weather_code"],
    })
    # Drop rows missing core training features only
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    df["rained"] = df["weather_code"].isin(RAIN_CODES).astype(int)
    return df


async def fetch_historical_air_quality(
    latitude: float,
    longitude: float,
    days: int = 90,
) -> pd.DataFrame:
    """Past N days of air quality aggregated to daily max.

    Columns: date, pm2_5, pm10, european_aqi
    """
    days = max(1, min(days, 92))
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "pm10,pm2_5,european_aqi",
        "past_days": days,
        "timezone": "auto",
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.get(AIR_QUALITY_URL, params=params)
        r.raise_for_status()
        data = r.json()
    return _parse_hourly_aq(data)


async def fetch_forecast_weather(
    latitude: float,
    longitude: float,
    days: int = 7,
) -> pd.DataFrame:
    """Daily forecast including UV, apparent temperature, and wind gusts.

    Columns: date, precipitation, temp_max, temp_min, windspeed,
             uv_index, apparent_temp_max, wind_gusts, weather_code
    """
    days = max(1, min(days, 16))
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ",".join([
            "weather_code",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
            "uv_index_max",
            "apparent_temperature_max",
            "wind_gusts_10m_max",
        ]),
        "forecast_days": days,
        "timezone": "auto",
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.get(FORECAST_URL, params=params)
        r.raise_for_status()
        data = r.json()

    daily = data["daily"]
    n = len(daily["time"])
    return pd.DataFrame({
        "date":             pd.to_datetime(daily["time"]),
        "precipitation":    daily["precipitation_sum"],
        "temp_max":         daily["temperature_2m_max"],
        "temp_min":         daily["temperature_2m_min"],
        "windspeed":        daily["wind_speed_10m_max"],
        "uv_index":         daily.get("uv_index_max") or [None] * n,
        "apparent_temp_max":daily.get("apparent_temperature_max") or [None] * n,
        "wind_gusts":       daily.get("wind_gusts_10m_max") or [None] * n,
        "weather_code":     daily["weather_code"],
    }).reset_index(drop=True)


async def fetch_air_quality(
    latitude: float,
    longitude: float,
    days: int = 7,
) -> pd.DataFrame:
    """Forecast air quality aggregated to daily max.

    Columns: date, pm2_5, pm10, european_aqi
    """
    days = max(1, min(days, 7))
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "pm10,pm2_5,european_aqi",
        "forecast_days": days,
        "timezone": "auto",
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.get(AIR_QUALITY_URL, params=params)
        r.raise_for_status()
        data = r.json()
    return _parse_hourly_aq(data)


def _parse_hourly_aq(data: dict) -> pd.DataFrame:
    hourly = data["hourly"]
    df = pd.DataFrame({
        "datetime":     pd.to_datetime(hourly["time"]),
        "pm2_5":        hourly.get("pm2_5") or [],
        "pm10":         hourly.get("pm10") or [],
        "european_aqi": hourly.get("european_aqi") or [],
    }).dropna()

    if df.empty:
        return pd.DataFrame(columns=["date", "pm2_5", "pm10", "european_aqi"])

    df["date"] = df["datetime"].dt.normalize()
    return (
        df.groupby("date")[["pm2_5", "pm10", "european_aqi"]]
        .max()
        .reset_index()
        .assign(date=lambda d: pd.to_datetime(d["date"]))
    )
