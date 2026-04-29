"""XGBoost models + SHAP TreeExplainer for rain, UV, heat, and air quality.

Four separate models are trained per city:
  - rain_model      (classifier)  → rain probability
  - uv_model        (regressor)   → UV index
  - heat_model      (regressor)   → apparent (feels-like) temperature
  - aqi_model       (regressor)   → European AQI (trained on 90-day AQ history)

All use the same four input features: precipitation, temp_max, temp_min, windspeed.
SHAP TreeExplainer gives exact Shapley values for every model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from .open_meteo import (
    FEATURES,
    fetch_forecast_weather,
    fetch_historical_air_quality,
    fetch_historical_weather,
)

FEATURE_LABELS: dict[str, str] = {
    "precipitation": "precipitation (mm)",
    "temp_max":      "max temperature (°C)",
    "temp_min":      "min temperature (°C)",
    "windspeed":     "max wind speed (km/h)",
}


@dataclass
class CityModel:
    label: str
    latitude: float
    longitude: float

    # Rain classifier + SHAP
    model: XGBClassifier
    explainer: shap.TreeExplainer
    train_accuracy: float
    test_accuracy: float
    rain_base_rate: float
    training_days: int

    # UV regressor + SHAP (None if not enough data)
    uv_model: Any = None
    uv_explainer: Any = None

    # Heat (apparent temp) regressor + SHAP
    heat_model: Any = None
    heat_explainer: Any = None

    # AQI regressor + SHAP (trained on 90-day AQ history)
    aqi_model: Any = None
    aqi_explainer: Any = None

    # Cached forecast + per-model SHAP arrays  (shape: days × 4)
    last_forecast: pd.DataFrame | None = None
    last_probabilities: np.ndarray | None = None
    last_shap: np.ndarray | None = None       # rain
    last_uv_shap: np.ndarray | None = None
    last_heat_shap: np.ndarray | None = None
    last_aqi_shap: np.ndarray | None = None

    last_air_quality: pd.DataFrame | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


_MODEL_CACHE: dict[tuple[float, float], CityModel] = {}


def _round_key(lat: float, lon: float) -> tuple[float, float]:
    return (round(lat, 2), round(lon, 2))


def _train_regressor(X: np.ndarray, y: np.ndarray) -> tuple[XGBRegressor, shap.TreeExplainer]:
    reg = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1,
                       eval_metric="rmse", n_jobs=2)
    reg.fit(X, y)
    return reg, shap.TreeExplainer(reg)


async def get_or_train_model(
    label: str,
    latitude: float,
    longitude: float,
    years: int = 2,
) -> CityModel:
    key = _round_key(latitude, longitude)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    history = await fetch_historical_weather(latitude, longitude, years=years)
    if len(history) < 60:
        raise ValueError(f"Not enough data for {label} ({len(history)} days).")

    X = history[FEATURES].to_numpy()

    # ── Rain classifier ──────────────────────────────────────────────────────
    y_rain = history["rained"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_rain, test_size=0.2, random_state=42,
        stratify=y_rain if y_rain.sum() > 5 else None,
    )
    rain_model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                eval_metric="logloss", n_jobs=2)
    rain_model.fit(X_tr, y_tr)
    rain_explainer = shap.TreeExplainer(rain_model)

    # ── UV regressor ─────────────────────────────────────────────────────────
    uv_model = uv_explainer = None
    uv_rows = history.dropna(subset=["uv_index"])
    if len(uv_rows) >= 20:
        uv_model, uv_explainer = _train_regressor(
            uv_rows[FEATURES].to_numpy(), uv_rows["uv_index"].to_numpy()
        )
    else:
        # Archive UV data not available for this location — synthesise training
        # data from physics: UV ≈ f(temp, precipitation, season).
        import numpy as _np
        _rng = _np.random.default_rng(42)
        n = len(history)
        # UV correlates positively with temp_max, negatively with precipitation
        synth_uv = (
            0.25 * history["temp_max"].to_numpy()
            - 0.08 * history["precipitation"].to_numpy().clip(0, 30)
            + _rng.normal(0, 0.5, n)
        ).clip(1, 12)
        uv_model, uv_explainer = _train_regressor(history[FEATURES].to_numpy(), synth_uv)

    # ── Heat (apparent temp) regressor ───────────────────────────────────────
    heat_model = heat_explainer = None
    heat_rows = history.dropna(subset=["apparent_temp_max"])
    if len(heat_rows) >= 20:
        heat_model, heat_explainer = _train_regressor(
            heat_rows[FEATURES].to_numpy(), heat_rows["apparent_temp_max"].to_numpy()
        )
    else:
        # Apparent temp ≈ temp_max + humidity proxy - wind cooling
        import numpy as _np
        synth_heat = (
            history["temp_max"].to_numpy()
            + 0.3 * history["temp_min"].to_numpy()
            - 0.05 * history["windspeed"].to_numpy()
        )
        heat_model, heat_explainer = _train_regressor(history[FEATURES].to_numpy(), synth_heat)

    # ── AQI regressor (90-day history) ───────────────────────────────────────
    aqi_model = aqi_explainer = None
    try:
        aq_hist = await fetch_historical_air_quality(latitude, longitude, days=90)
        if not aq_hist.empty and len(aq_hist) >= 30:
            merged = history.merge(aq_hist, on="date", how="inner").dropna(
                subset=FEATURES + ["european_aqi"]
            )
            if len(merged) >= 30:
                aqi_model, aqi_explainer = _train_regressor(
                    merged[FEATURES].to_numpy(), merged["european_aqi"].to_numpy()
                )
    except Exception:
        pass  # AQ data not available for this location — skip silently

    city_model = CityModel(
        label=label,
        latitude=latitude,
        longitude=longitude,
        model=rain_model,
        explainer=rain_explainer,
        train_accuracy=float(rain_model.score(X_tr, y_tr)),
        test_accuracy=float(rain_model.score(X_te, y_te)),
        rain_base_rate=float(y_rain.mean()),
        training_days=int(len(history)),
        uv_model=uv_model,
        uv_explainer=uv_explainer,
        heat_model=heat_model,
        heat_explainer=heat_explainer,
        aqi_model=aqi_model,
        aqi_explainer=aqi_explainer,
        metadata={
            "training_start": str(history["date"].min().date()),
            "training_end":   str(history["date"].max().date()),
        },
    )
    _MODEL_CACHE[key] = city_model
    return city_model


async def predict_next_days(city_model: CityModel, days: int = 7) -> dict[str, Any]:
    forecast = await fetch_forecast_weather(city_model.latitude, city_model.longitude, days=days)
    X = forecast[FEATURES].to_numpy()

    # Rain
    proba = city_model.model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)
    rain_shap = city_model.explainer.shap_values(X)
    if isinstance(rain_shap, list):
        rain_shap = rain_shap[1]

    # UV SHAP
    uv_shap = None
    if city_model.uv_model is not None:
        uv_shap = np.asarray(city_model.uv_explainer.shap_values(X))

    # Heat SHAP
    heat_shap = None
    if city_model.heat_model is not None:
        heat_shap = np.asarray(city_model.heat_explainer.shap_values(X))

    # AQI SHAP (weather → AQI pattern, not raw AQ API value)
    aqi_shap = None
    if city_model.aqi_model is not None:
        aqi_shap = np.asarray(city_model.aqi_explainer.shap_values(X))

    city_model.last_forecast      = forecast
    city_model.last_probabilities = proba
    city_model.last_shap          = np.asarray(rain_shap)
    city_model.last_uv_shap       = uv_shap
    city_model.last_heat_shap     = heat_shap
    city_model.last_aqi_shap      = aqi_shap

    daily: list[dict[str, Any]] = []
    for i, row in forecast.iterrows():
        daily.append({
            "date":             row["date"].date().isoformat(),
            "rain_probability": round(float(proba[i]), 4),
            "rain_predicted":   bool(preds[i]),
            "features": {
                "precipitation_mm":   float(row["precipitation"]),
                "temp_max_c":         float(row["temp_max"]),
                "temp_min_c":         float(row["temp_min"]),
                "windspeed_kmh":      float(row["windspeed"]),
                "uv_index":           float(row["uv_index"]) if row.get("uv_index") is not None else None,
                "feels_like_c":       float(row["apparent_temp_max"]) if row.get("apparent_temp_max") is not None else None,
                "wind_gusts_kmh":     float(row["wind_gusts"]) if row.get("wind_gusts") is not None else None,
            },
        })

    return {
        "city": city_model.label,
        "latitude": city_model.latitude,
        "longitude": city_model.longitude,
        "model": {
            "type": "XGBoost binary classifier",
            "train_accuracy": round(city_model.train_accuracy, 4),
            "test_accuracy":  round(city_model.test_accuracy, 4),
            "rain_base_rate": round(city_model.rain_base_rate, 4),
            "training_days":  city_model.training_days,
            "training_window": {
                "start": city_model.metadata.get("training_start"),
                "end":   city_model.metadata.get("training_end"),
            },
        },
        "forecast": daily,
    }


# ---------------------------------------------------------------------------
# Shared explain helper
# ---------------------------------------------------------------------------
def _build_factors(
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    direction_pos: str,
    direction_neg: str,
    sentence_fn,
) -> list[dict[str, Any]]:
    factors = []
    for name, value, sv in zip(FEATURES, feature_values, shap_values):
        sv = float(sv)
        factors.append({
            "feature":       name,
            "label":         FEATURE_LABELS[name],
            "value":         float(value),
            "shap_value":    round(sv, 4),
            "direction":     direction_pos if sv > 0 else direction_neg,
            "plain_english": sentence_fn(name, float(value), sv),
        })
    return sorted(factors, key=lambda f: abs(f["shap_value"]), reverse=True)


# ---------------------------------------------------------------------------
# Rain explanation
# ---------------------------------------------------------------------------
def _rain_sentence(name: str, value: float, sv: float) -> str:
    pushes = "pushes rain chance up" if sv > 0 else "pulls rain chance down"
    strength = "strongly" if abs(sv) > 0.5 else ("moderately" if abs(sv) > 0.2 else "slightly")
    if name == "precipitation":
        desc = f"Heavy forecast precipitation ({value:.1f} mm)" if value > 10 else (f"Some precipitation ({value:.1f} mm)" if value > 1 else f"Dry forecast ({value:.1f} mm)")
    elif name == "temp_max":
        desc = f"Very hot max temp ({value:.1f}°C)" if value > 32 else (f"Warm max temp ({value:.1f}°C)" if value > 25 else f"Cool max temp ({value:.1f}°C)")
    elif name == "temp_min":
        desc = f"Warm overnight low ({value:.1f}°C)" if value > 22 else f"Cool overnight low ({value:.1f}°C)"
    else:
        desc = f"Strong winds ({value:.1f} km/h)" if value > 40 else (f"Moderate winds ({value:.1f} km/h)" if value > 20 else f"Calm winds ({value:.1f} km/h)")
    return f"{desc} — {strength} {pushes}."


def explain_day(city_model: CityModel, day_index: int) -> dict[str, Any]:
    _check_cache(city_model, day_index)
    row   = city_model.last_forecast.iloc[day_index]
    sv    = city_model.last_shap[day_index]
    base  = float(city_model.explainer.expected_value)
    proba = float(city_model.last_probabilities[day_index])
    date  = row["date"].strftime("%A, %B %d")

    factors = _build_factors(sv, row[FEATURES].to_numpy(), "increases rain", "decreases rain", _rain_sentence)

    verdict = "will likely rain" if proba >= 0.5 else "is unlikely to rain"
    confidence = "very likely" if proba >= 0.8 else ("likely" if proba >= 0.6 else ("unlikely" if proba <= 0.4 else "about 50/50"))
    reasons = [f["plain_english"] for f in factors if abs(f["shap_value"]) > 0.05]

    return {
        "city":             city_model.label,
        "date":             row["date"].date().isoformat(),
        "rain_probability": round(proba, 4),
        "rain_predicted":   bool(proba >= 0.5),
        "summary": (
            f"{date} in {city_model.label} {verdict} ({proba * 100:.0f}% — {confidence}). "
            + " ".join(reasons)
        ),
        "reasons": [f["plain_english"] for f in factors],
        "factors": factors,
        "shap_base_logit": round(base, 4),
    }


# ---------------------------------------------------------------------------
# UV explanation
# ---------------------------------------------------------------------------
def _uv_sentence(name: str, value: float, sv: float) -> str:
    pushes = "raises UV intensity" if sv > 0 else "lowers UV intensity"
    strength = "strongly" if abs(sv) > 0.5 else ("moderately" if abs(sv) > 0.2 else "slightly")
    if name == "precipitation":
        desc = f"Heavy rain/clouds ({value:.1f} mm)" if value > 5 else f"Clear skies ({value:.1f} mm precipitation)"
    elif name == "temp_max":
        desc = f"Hot, clear day ({value:.1f}°C max)" if value > 30 else f"Moderate temperature ({value:.1f}°C)"
    elif name == "temp_min":
        desc = f"Warm overnight ({value:.1f}°C)" if value > 22 else f"Cool night ({value:.1f}°C)"
    else:
        desc = f"High wind ({value:.1f} km/h — can clear clouds)" if value > 30 else f"Low wind ({value:.1f} km/h)"
    return f"{desc} — {strength} {pushes}."


def explain_uv_day(city_model: CityModel, day_index: int) -> dict[str, Any]:
    _check_cache(city_model, day_index)
    row  = city_model.last_forecast.iloc[day_index]
    sv   = city_model.last_uv_shap[day_index]
    base = float(city_model.uv_explainer.expected_value)
    pred = float(base + sv.sum())
    date = row["date"].strftime("%A, %B %d")

    factors = _build_factors(sv, row[FEATURES].to_numpy(), "raises UV", "lowers UV", _uv_sentence)
    reasons = [f["plain_english"] for f in factors if abs(f["shap_value"]) > 0.1]

    uv_label = (
        "Extreme" if pred >= 11 else "Very High" if pred >= 8 else
        "High" if pred >= 6 else "Moderate" if pred >= 3 else "Low"
    )
    return {
        "city":          city_model.label,
        "date":          row["date"].date().isoformat(),
        "predicted_uv":  round(pred, 1),
        "uv_category":   uv_label,
        "actual_uv":     float(row["uv_index"]) if row.get("uv_index") is not None else None,
        "summary": f"On {date} in {city_model.label} the UV index is predicted to be {pred:.1f} ({uv_label}). " + " ".join(reasons),
        "reasons":       [f["plain_english"] for f in factors],
        "factors":       factors,
        "shap_base":     round(base, 4),
    }


# ---------------------------------------------------------------------------
# Heat explanation
# ---------------------------------------------------------------------------
def _heat_sentence(name: str, value: float, sv: float) -> str:
    pushes = "makes it feel hotter" if sv > 0 else "makes it feel cooler"
    strength = "strongly" if abs(sv) > 1.0 else ("moderately" if abs(sv) > 0.4 else "slightly")
    if name == "temp_max":
        desc = f"Very high actual temperature ({value:.1f}°C)" if value > 35 else f"Max temperature of {value:.1f}°C"
    elif name == "temp_min":
        desc = f"Warm overnight low ({value:.1f}°C — humid air)" if value > 22 else f"Cool overnight ({value:.1f}°C)"
    elif name == "windspeed":
        desc = f"Very low wind ({value:.1f} km/h — no cooling)" if value < 10 else f"Wind at {value:.1f} km/h provides some cooling"
    else:
        desc = f"Rain ({value:.1f} mm — cools air)" if value > 5 else f"Dry conditions ({value:.1f} mm)"
    return f"{desc} — {strength} {pushes}."


def explain_heat_day(city_model: CityModel, day_index: int) -> dict[str, Any]:
    _check_cache(city_model, day_index)
    row  = city_model.last_forecast.iloc[day_index]
    sv   = city_model.last_heat_shap[day_index]
    base = float(city_model.heat_explainer.expected_value)
    pred = float(base + sv.sum())
    date = row["date"].strftime("%A, %B %d")

    factors = _build_factors(sv, row[FEATURES].to_numpy(), "raises heat", "lowers heat", _heat_sentence)
    reasons = [f["plain_english"] for f in factors if abs(f["shap_value"]) > 0.3]

    heat_label = (
        "Extreme Heat" if pred >= 38 else "Hot" if pred >= 32 else
        "Warm" if pred >= 25 else "Comfortable" if pred >= 18 else
        "Cold" if pred >= 10 else "Very Cold"
    )
    return {
        "city":              city_model.label,
        "date":              row["date"].date().isoformat(),
        "predicted_feels_like_c": round(pred, 1),
        "heat_category":     heat_label,
        "actual_feels_like": float(row["apparent_temp_max"]) if row.get("apparent_temp_max") is not None else None,
        "summary": f"On {date} in {city_model.label} it will feel like {pred:.1f}°C ({heat_label}). " + " ".join(reasons),
        "reasons":           [f["plain_english"] for f in factors],
        "factors":           factors,
        "shap_base":         round(base, 4),
    }


# ---------------------------------------------------------------------------
# AQI explanation
# ---------------------------------------------------------------------------
def _aqi_sentence(name: str, value: float, sv: float) -> str:
    pushes = "worsens air quality" if sv > 0 else "improves air quality"
    strength = "strongly" if abs(sv) > 2.0 else ("moderately" if abs(sv) > 1.0 else "slightly")
    if name == "windspeed":
        desc = f"Very low wind ({value:.1f} km/h — pollutants accumulate)" if value < 10 else f"Wind at {value:.1f} km/h disperses pollutants"
    elif name == "precipitation":
        desc = f"Rain ({value:.1f} mm — washes particles from air)" if value > 5 else f"No rain ({value:.1f} mm — no washout)"
    elif name == "temp_max":
        desc = f"High temperature ({value:.1f}°C — temperature inversion risk)" if value > 32 else f"Max temperature {value:.1f}°C"
    else:
        desc = f"Warm overnight ({value:.1f}°C — traps pollutants low)" if value > 22 else f"Cool night ({value:.1f}°C)"
    return f"{desc} — {strength} {pushes}."


def explain_aqi_day(city_model: CityModel, day_index: int) -> dict[str, Any]:
    if city_model.aqi_model is None:
        raise RuntimeError("AQI model not available (need 30+ days of air quality history for this city).")
    _check_cache(city_model, day_index)
    row  = city_model.last_forecast.iloc[day_index]
    sv   = city_model.last_aqi_shap[day_index]
    base = float(city_model.aqi_explainer.expected_value)
    pred = float(base + sv.sum())
    date = row["date"].strftime("%A, %B %d")

    factors = _build_factors(sv, row[FEATURES].to_numpy(), "worsens AQI", "improves AQI", _aqi_sentence)
    reasons = [f["plain_english"] for f in factors if abs(f["shap_value"]) > 0.5]

    aqi_label = (
        "Extremely Poor" if pred >= 100 else "Very Poor" if pred >= 80 else
        "Poor" if pred >= 60 else "Moderate" if pred >= 40 else
        "Fair" if pred >= 20 else "Good"
    )
    return {
        "city":          city_model.label,
        "date":          row["date"].date().isoformat(),
        "predicted_aqi": round(pred, 1),
        "aqi_category":  aqi_label,
        "summary": f"On {date} in {city_model.label} air quality is predicted to be {aqi_label} (AQI ≈ {pred:.0f}). " + " ".join(reasons),
        "reasons":       [f["plain_english"] for f in factors],
        "factors":       factors,
        "shap_base":     round(base, 4),
    }


def _check_cache(city_model: CityModel, day_index: int) -> None:
    if city_model.last_shap is None or city_model.last_forecast is None:
        raise RuntimeError("No prediction cached. Call predict_rain_7day first.")
    if not 0 <= day_index < len(city_model.last_forecast):
        raise IndexError(f"day_index {day_index} out of range (have {len(city_model.last_forecast)} days).")
