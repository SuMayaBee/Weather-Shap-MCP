"""Weather MCP server (FastMCP) — rain, UV, air quality, activity scoring."""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP, Image

from .model import (
    explain_aqi_day,
    explain_day,
    explain_heat_day,
    explain_uv_day,
    get_or_train_model,
    predict_next_days,
)
from .open_meteo import fetch_air_quality, geocode_city
from .scoring import (
    air_quality_summary,
    find_best_day,
    score_all_days,
    uv_heat_summary,
)
from .visualize import (
    activity_score_chart_png,
    air_quality_chart_png,
    aqi_shap_waterfall_png,
    forecast_chart_png,
    heat_shap_waterfall_png,
    shap_waterfall_png,
    uv_heat_chart_png,
    uv_shap_waterfall_png,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("weather-mcp")

mcp = FastMCP("weather-mcp")


async def _ensure_forecast(model, days: int = 7):
    if model.last_forecast is None:
        await predict_next_days(model, days=days)


# ---------------------------------------------------------------------------
# Tool 1 — Rain forecast
# ---------------------------------------------------------------------------
@mcp.tool()
async def predict_rain_7day(city: str, days: int = 7) -> list:
    """Predict rain probability for the next N days in any city.

    Returns a bar chart + daily data. Trains an XGBoost model on 2 years of
    Open-Meteo archive data on first call, then caches it.

    Args:
        city: City name, e.g. 'Tokyo' or 'Dhaka, Bangladesh'.
        days: Number of forecast days (1-16, default 7).
    """
    days = max(1, min(int(days), 16))
    geo = await geocode_city(city)
    model = await get_or_train_model(geo["label"], geo["latitude"], geo["longitude"])
    result = await predict_next_days(model, days=days)
    png = forecast_chart_png(model)
    return [json.dumps(result, indent=2, default=str), Image(data=png, format="png")]


# ---------------------------------------------------------------------------
# Tool 2 — SHAP explanation
# ---------------------------------------------------------------------------
@mcp.tool()
async def explain_rain_prediction(city: str, day_index: int = 0) -> list:
    """Explain WHY rain is (or isn't) predicted for a specific day using SHAP.

    Returns a SHAP waterfall plot + plain-English reasons. Runs a 7-day
    forecast first if none is cached for this city.

    Args:
        city: City name.
        day_index: 0 = tomorrow, 1 = day after, etc.
    """
    geo = await geocode_city(city)
    model = await get_or_train_model(geo["label"], geo["latitude"], geo["longitude"])
    await _ensure_forecast(model)
    result = explain_day(model, int(day_index))
    png = shap_waterfall_png(model, int(day_index))
    return [json.dumps(result, indent=2, default=str), Image(data=png, format="png")]


# ---------------------------------------------------------------------------
# Tool 3 — Outdoor activity score
# ---------------------------------------------------------------------------
@mcp.tool()
async def get_outdoor_activity_score(
    city: str,
    activity: str = "general",
    days: int = 7,
) -> list:
    """Score each day 0-100 for outdoor activities based on rain, UV, heat, and wind.

    Returns a colour-coded bar chart + day-by-day breakdown with reasons.

    Args:
        city: City name.
        activity: One of general, running, cycling, beach, picnic, event.
        days: Number of forecast days (1-16, default 7).
    """
    days = max(1, min(int(days), 16))
    geo = await geocode_city(city)
    model = await get_or_train_model(geo["label"], geo["latitude"], geo["longitude"])
    await _ensure_forecast(model, days)
    scored = score_all_days(model.last_forecast, model.last_probabilities, activity)
    png = activity_score_chart_png(model.label, scored)
    result = {
        "city": model.label,
        "activity": activity,
        "days": scored,
        "tip": f"Best day: {max(scored, key=lambda d: d['score'])['day_name']} "
               f"({max(scored, key=lambda d: d['score'])['score']}/100)",
    }
    return [json.dumps(result, indent=2, default=str), Image(data=png, format="png")]


# ---------------------------------------------------------------------------
# Tool 4 — UV index + heatwave forecast
# ---------------------------------------------------------------------------
@mcp.tool()
async def get_uv_and_heat_forecast(city: str, days: int = 7) -> list:
    """Forecast UV index and feels-like temperature for the next N days.

    Returns a dual-panel chart (UV bars + heat line) + per-day categories
    and advice (e.g. 'Extreme — stay indoors during peak hours').

    Args:
        city: City name.
        days: Number of forecast days (1-16, default 7).
    """
    days = max(1, min(int(days), 16))
    geo = await geocode_city(city)
    model = await get_or_train_model(geo["label"], geo["latitude"], geo["longitude"])
    await _ensure_forecast(model, days)
    summary = uv_heat_summary(model.last_forecast)
    png = uv_heat_chart_png(model.label, summary)
    result = {"city": model.label, "days": summary}
    return [json.dumps(result, indent=2, default=str), Image(data=png, format="png")]


# ---------------------------------------------------------------------------
# Tool 5 — Air quality
# ---------------------------------------------------------------------------
@mcp.tool()
async def get_air_quality(city: str, days: int = 7) -> list:
    """Forecast air quality (PM2.5, PM10, European AQI) for the next N days.

    Returns a dual-panel chart (AQI bars + particulate lines with WHO limits)
    + per-day categories and health advice.

    Args:
        city: City name.
        days: Number of forecast days (1-7, default 7).
    """
    days = max(1, min(int(days), 7))
    geo = await geocode_city(city)
    model = await get_or_train_model(geo["label"], geo["latitude"], geo["longitude"])
    aq_df = await fetch_air_quality(geo["latitude"], geo["longitude"], days=days)
    model.last_air_quality = aq_df
    summary = air_quality_summary(aq_df)
    png = air_quality_chart_png(model.label, summary)
    result = {"city": model.label, "days": summary}
    return [json.dumps(result, indent=2, default=str), Image(data=png, format="png")]


# ---------------------------------------------------------------------------
# Tool 5b — Explain UV prediction (SHAP)
# ---------------------------------------------------------------------------
@mcp.tool()
async def explain_uv_prediction(city: str, day_index: int = 0) -> list:
    """Explain WHY the UV index is predicted high or low for a specific day using SHAP.

    Returns a SHAP waterfall plot showing which weather conditions drive UV intensity,
    plus a plain-English explanation (e.g. 'Clear skies with no rain strongly raises UV').

    Args:
        city: City name.
        day_index: 0 = tomorrow, 1 = day after, etc.
    """
    geo = await geocode_city(city)
    model = await get_or_train_model(geo["label"], geo["latitude"], geo["longitude"])
    await _ensure_forecast(model)
    result = explain_uv_day(model, int(day_index))
    png = uv_shap_waterfall_png(model, int(day_index))
    return [json.dumps(result, indent=2, default=str), Image(data=png, format="png")]


# ---------------------------------------------------------------------------
# Tool 5c — Explain heat prediction (SHAP)
# ---------------------------------------------------------------------------
@mcp.tool()
async def explain_heat_prediction(city: str, day_index: int = 0) -> list:
    """Explain WHY it will feel so hot (or cool) on a specific day using SHAP.

    Returns a SHAP waterfall plot showing which factors drive the feels-like temperature,
    plus plain-English reasons (e.g. 'Very low wind provides no cooling — strongly makes it feel hotter').

    Args:
        city: City name.
        day_index: 0 = tomorrow, 1 = day after, etc.
    """
    geo = await geocode_city(city)
    model = await get_or_train_model(geo["label"], geo["latitude"], geo["longitude"])
    await _ensure_forecast(model)
    result = explain_heat_day(model, int(day_index))
    png = heat_shap_waterfall_png(model, int(day_index))
    return [json.dumps(result, indent=2, default=str), Image(data=png, format="png")]


# ---------------------------------------------------------------------------
# Tool 5d — Explain air quality prediction (SHAP)
# ---------------------------------------------------------------------------
@mcp.tool()
async def explain_air_quality_prediction(city: str, day_index: int = 0) -> list:
    """Explain WHY air quality is predicted good or poor for a specific day using SHAP.

    Trained on 90 days of historical AQ data — SHAP shows which weather patterns
    (low wind, no rain, high temp) are driving pollution levels.

    Args:
        city: City name.
        day_index: 0 = tomorrow, 1 = day after, etc.
    """
    geo = await geocode_city(city)
    model = await get_or_train_model(geo["label"], geo["latitude"], geo["longitude"])
    await _ensure_forecast(model)
    result = explain_aqi_day(model, int(day_index))
    png = aqi_shap_waterfall_png(model, int(day_index))
    return [json.dumps(result, indent=2, default=str), Image(data=png, format="png")]


# ---------------------------------------------------------------------------
# Tool 6 — Best day finder
# ---------------------------------------------------------------------------
@mcp.tool()
async def find_best_day_for(
    city: str,
    activity: str = "general",
    days: int = 7,
) -> dict[str, Any]:
    """Find the single best day this week for an outdoor activity.

    Ranks all days by a composite score (rain probability + UV + heat + wind)
    and returns the winner with a plain-English explanation.

    Args:
        city: City name.
        activity: One of general, running, cycling, beach, picnic, event.
        days: Number of forecast days to consider (1-16, default 7).
    """
    days = max(1, min(int(days), 16))
    geo = await geocode_city(city)
    model = await get_or_train_model(geo["label"], geo["latitude"], geo["longitude"])
    await _ensure_forecast(model, days)
    scored = score_all_days(model.last_forecast, model.last_probabilities, activity)
    return find_best_day(scored, activity)


# ---------------------------------------------------------------------------
# Tool 7 — Model info
# ---------------------------------------------------------------------------
@mcp.tool()
async def city_model_info(city: str) -> dict[str, Any]:
    """Return metadata about the trained rain model for a city.

    Args:
        city: City name.
    """
    geo = await geocode_city(city)
    model = await get_or_train_model(geo["label"], geo["latitude"], geo["longitude"])
    return {
        "city": model.label,
        "latitude": model.latitude,
        "longitude": model.longitude,
        "train_accuracy": round(model.train_accuracy, 4),
        "test_accuracy": round(model.test_accuracy, 4),
        "rain_base_rate": round(model.rain_base_rate, 4),
        "training_days": model.training_days,
        "training_window": {
            "start": model.metadata.get("training_start"),
            "end": model.metadata.get("training_end"),
        },
        "has_cached_forecast": model.last_forecast is not None,
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
