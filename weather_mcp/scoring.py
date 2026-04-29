"""Rule-based scoring for weather conditions — no ML model needed.

All functions work from the forecast DataFrame already stored on CityModel,
plus air quality data fetched separately.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# UV index bands (WHO standard)
# ---------------------------------------------------------------------------
_UV_BANDS = [
    (0,  2,  "Low",       "#4CAF50", "No protection needed."),
    (3,  5,  "Moderate",  "#FFEB3B", "Wear sunscreen SPF 30+."),
    (6,  7,  "High",      "#FF9800", "Hat + sunscreen essential."),
    (8,  10, "Very High", "#F44336", "Avoid midday sun."),
    (11, 99, "Extreme",   "#9C27B0", "Stay indoors during peak hours."),
]

# ---------------------------------------------------------------------------
# European AQI bands
# ---------------------------------------------------------------------------
_AQI_BANDS = [
    (0,   20,  "Good",           "#4CAF50", "Air quality is excellent."),
    (20,  40,  "Fair",           "#8BC34A", "Air quality is acceptable."),
    (40,  60,  "Moderate",       "#FFEB3B", "Sensitive groups should reduce outdoor time."),
    (60,  80,  "Poor",           "#FF9800", "Everyone may experience effects."),
    (80,  100, "Very Poor",      "#F44336", "Avoid prolonged outdoor activity."),
    (100, 999, "Extremely Poor", "#9C27B0", "Stay indoors."),
]

# ---------------------------------------------------------------------------
# Heat bands (apparent / feels-like temperature)
# ---------------------------------------------------------------------------
_HEAT_BANDS = [
    (-99, 0,  "Freezing",      "🧊", "Layer up — dangerous cold possible."),
    (0,   10, "Very Cold",     "🥶", "Heavy coat required."),
    (10,  18, "Cold",          "🧥", "Wear a jacket."),
    (18,  25, "Comfortable",   "😊", "Pleasant conditions."),
    (25,  32, "Warm",          "🌤",  "Light clothing suitable."),
    (32,  38, "Hot",           "🥵", "Stay hydrated, limit exertion."),
    (38,  99, "Extreme Heat",  "🔥", "Dangerous — avoid outdoor activity."),
]


def _band(value: float, bands: list) -> dict[str, Any]:
    for lo, hi, label, color, advice in bands:
        if lo <= value <= hi:
            return {"label": label, "color": color, "advice": advice}
    return {"label": "Unknown", "color": "#9E9E9E", "advice": ""}


# ---------------------------------------------------------------------------
# Outdoor activity score (0-100)
# ---------------------------------------------------------------------------
_ACTIVITY_WEIGHTS = {
    "general":  {"rain": 35, "uv": 20, "heat": 25, "wind": 20},
    "running":  {"rain": 30, "uv": 15, "heat": 30, "wind": 25},
    "cycling":  {"rain": 30, "uv": 10, "heat": 25, "wind": 35},
    "beach":    {"rain": 40, "uv":  5, "heat": 10, "wind": 45},
    "picnic":   {"rain": 40, "uv": 20, "heat": 25, "wind": 15},
    "event":    {"rain": 50, "uv": 15, "heat": 20, "wind": 15},
}


def activity_score_for_day(
    rain_prob: float,
    uv: float | None,
    apparent_temp: float | None,
    wind_gusts: float | None,
    activity: str = "general",
) -> dict[str, Any]:
    """Return a 0-100 score and breakdown for one day."""
    w = _ACTIVITY_WEIGHTS.get(activity, _ACTIVITY_WEIGHTS["general"])

    rain_penalty  = rain_prob * w["rain"]
    uv_penalty    = min(max(0, ((uv or 0) - 6) * 4), w["uv"])
    heat_penalty  = min(max(0, ((apparent_temp or 20) - 33) * 2), w["heat"])
    wind_penalty  = min(max(0, ((wind_gusts or 0) - 35) * 0.6), w["wind"])

    score = max(0, min(100, round(100 - rain_penalty - uv_penalty - heat_penalty - wind_penalty)))

    if score >= 80:
        verdict, emoji = "Excellent", "🌟"
    elif score >= 65:
        verdict, emoji = "Good", "✅"
    elif score >= 45:
        verdict, emoji = "Fair", "⚠️"
    elif score >= 25:
        verdict, emoji = "Poor", "❌"
    else:
        verdict, emoji = "Bad", "🚫"

    reasons = []
    if rain_prob >= 0.6:
        reasons.append(f"High rain chance ({rain_prob*100:.0f}%)")
    if uv is not None and uv >= 6:
        reasons.append(f"UV is {_band(uv, _UV_BANDS)['label']} ({uv:.0f})")
    if apparent_temp is not None and apparent_temp >= 33:
        reasons.append(f"Feels like {apparent_temp:.0f}°C — very hot")
    if wind_gusts is not None and wind_gusts >= 35:
        reasons.append(f"Strong wind gusts ({wind_gusts:.0f} km/h)")
    if not reasons:
        reasons.append("Conditions look fine across all factors")

    return {
        "score": score,
        "verdict": verdict,
        "emoji": emoji,
        "reasons": reasons,
    }


def score_all_days(
    forecast: pd.DataFrame,
    probabilities: "np.ndarray",
    activity: str = "general",
) -> list[dict[str, Any]]:
    """Score every forecasted day and return a ranked list."""
    rows = []
    for i, row in forecast.iterrows():
        s = activity_score_for_day(
            rain_prob=float(probabilities[i]),
            uv=row.get("uv_index"),
            apparent_temp=row.get("apparent_temp_max"),
            wind_gusts=row.get("wind_gusts"),
            activity=activity,
        )
        rows.append({
            "date": row["date"].date().isoformat(),
            "day_name": row["date"].strftime("%A"),
            **s,
        })
    return rows


# ---------------------------------------------------------------------------
# UV + heatwave summary per day
# ---------------------------------------------------------------------------
def uv_heat_summary(forecast: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for _, row in forecast.iterrows():
        uv = row.get("uv_index")
        at = row.get("apparent_temp_max")
        uv_info   = _band(uv, _UV_BANDS) if uv is not None else {"label": "N/A", "color": "#9E9E9E", "advice": ""}
        heat_info = _band(at, _HEAT_BANDS) if at is not None else {"label": "N/A", "color": "#9E9E9E", "advice": ""}
        rows.append({
            "date":            row["date"].date().isoformat(),
            "day_name":        row["date"].strftime("%A"),
            "uv_index":        float(uv) if uv is not None else None,
            "uv_category":     uv_info["label"],
            "uv_advice":       uv_info["advice"],
            "uv_color":        uv_info["color"],
            "feels_like_c":    float(at) if at is not None else None,
            "heat_category":   heat_info["label"],
            "heat_advice":     heat_info["advice"],
            "heat_emoji":      heat_info.get("emoji", ""),
        })
    return rows


# ---------------------------------------------------------------------------
# Air quality summary per day
# ---------------------------------------------------------------------------
def air_quality_summary(aq_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for _, row in aq_df.iterrows():
        aqi = row.get("european_aqi")
        info = _band(aqi, _AQI_BANDS) if aqi is not None else {"label": "N/A", "color": "#9E9E9E", "advice": ""}
        rows.append({
            "date":        row["date"].date().isoformat() if hasattr(row["date"], "date") else str(row["date"]),
            "pm2_5":       round(float(row["pm2_5"]), 1) if row.get("pm2_5") is not None else None,
            "pm10":        round(float(row["pm10"]), 1) if row.get("pm10") is not None else None,
            "european_aqi":round(float(aqi), 1) if aqi is not None else None,
            "aqi_category":info["label"],
            "aqi_color":   info["color"],
            "aqi_advice":  info["advice"],
        })
    return rows


# ---------------------------------------------------------------------------
# Best day finder
# ---------------------------------------------------------------------------
def find_best_day(
    scored_days: list[dict[str, Any]],
    activity: str = "general",
) -> dict[str, Any]:
    best = max(scored_days, key=lambda d: d["score"])
    return {
        "activity": activity,
        "best_day": best["day_name"],
        "date": best["date"],
        "score": best["score"],
        "verdict": best["verdict"],
        "emoji": best["emoji"],
        "reasons": best["reasons"],
        "all_days_ranked": sorted(scored_days, key=lambda d: d["score"], reverse=True),
    }
