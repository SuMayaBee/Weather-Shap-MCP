"""Generate PNG plots returned as bytes to the MCP client."""

from __future__ import annotations

import io
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import shap

from .model import CityModel
from .open_meteo import FEATURES

_FEATURE_LABELS = [
    "precipitation (mm)",
    "max temp (°C)",
    "min temp (°C)",
    "wind speed (km/h)",
]

_BG       = "#1a1a2e"
_PANEL    = "#16213e"
_BORDER   = "#0f3460"
_TEXT     = "white"
_SUBTEXT  = "#90CAF9"

# Use a consistent wide size so charts fill the chat window properly
_W, _H = 14, 6


def _save(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _dark_fig(w: float = _W, h: float = _H) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors=_SUBTEXT, labelsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor(_BORDER)
    ax.grid(axis="y", color=_BORDER, alpha=0.5, zorder=1)
    return fig, ax


# ---------------------------------------------------------------------------
# 1. Rain forecast bar chart
# ---------------------------------------------------------------------------
def forecast_chart_png(city_model: CityModel) -> bytes:
    forecast = city_model.last_forecast
    probs    = city_model.last_probabilities
    labels   = [r["date"].strftime("%b %d") for _, r in forecast.iterrows()]
    pct      = [p * 100 for p in probs]
    colors   = ["#1565C0" if p >= 50 else "#90CAF9" for p in pct]

    fig, ax = _dark_fig(10, 5)
    bars = ax.bar(labels, pct, color=colors, edgecolor=_BORDER, linewidth=0.8, zorder=3)
    for bar, p in zip(bars, pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{p:.0f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=_TEXT)

    ax.axhline(50, color="#ef5350", linestyle="--", linewidth=1.2, alpha=0.8,
               label="50% threshold", zorder=2)
    ax.set_ylim(0, 115)
    ax.set_xlabel("Date", color=_SUBTEXT, fontsize=11)
    ax.set_ylabel("Rain Probability (%)", color=_SUBTEXT, fontsize=11)
    ax.set_title(f"7-Day Rain Forecast — {city_model.label}",
                 color=_TEXT, fontsize=14, fontweight="bold", pad=12)
    ax.legend(facecolor=_BORDER, edgecolor=_BORDER, labelcolor=_TEXT, fontsize=9)
    return _save(fig)


# ---------------------------------------------------------------------------
# 2. Generic SHAP waterfall (reused for rain, UV, heat, AQI)
# ---------------------------------------------------------------------------
def _shap_waterfall_png(
    shap_values: "np.ndarray",
    base_value: float,
    feature_values: "np.ndarray",
    title: str,
) -> bytes:
    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=feature_values,
        feature_names=_FEATURE_LABELS,
    )
    shap.plots.waterfall(explanation, show=False)
    # SHAP creates its own figure — resize it after the fact
    fig = plt.gcf()
    fig.set_size_inches(_W, 5)
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def shap_waterfall_png(city_model: CityModel, day_index: int) -> bytes:
    row  = city_model.last_forecast.iloc[day_index]
    date = row["date"].strftime("%b %d, %Y")
    return _shap_waterfall_png(
        city_model.last_shap[day_index],
        float(city_model.explainer.expected_value),
        row[FEATURES].to_numpy(),
        f"SHAP — Rain Prediction · {city_model.label} ({date})",
    )


def uv_shap_waterfall_png(city_model: CityModel, day_index: int) -> bytes:
    row  = city_model.last_forecast.iloc[day_index]
    date = row["date"].strftime("%b %d, %Y")
    return _shap_waterfall_png(
        city_model.last_uv_shap[day_index],
        float(city_model.uv_explainer.expected_value),
        row[FEATURES].to_numpy(),
        f"SHAP — UV Index Prediction · {city_model.label} ({date})",
    )


def heat_shap_waterfall_png(city_model: CityModel, day_index: int) -> bytes:
    row  = city_model.last_forecast.iloc[day_index]
    date = row["date"].strftime("%b %d, %Y")
    return _shap_waterfall_png(
        city_model.last_heat_shap[day_index],
        float(city_model.heat_explainer.expected_value),
        row[FEATURES].to_numpy(),
        f"SHAP — Feels-Like Temperature · {city_model.label} ({date})",
    )


def aqi_shap_waterfall_png(city_model: CityModel, day_index: int) -> bytes:
    row  = city_model.last_forecast.iloc[day_index]
    date = row["date"].strftime("%b %d, %Y")
    return _shap_waterfall_png(
        city_model.last_aqi_shap[day_index],
        float(city_model.aqi_explainer.expected_value),
        row[FEATURES].to_numpy(),
        f"SHAP — Air Quality (AQI) Prediction · {city_model.label} ({date})",
    )


# ---------------------------------------------------------------------------
# 3. Outdoor activity score chart
# ---------------------------------------------------------------------------
_SCORE_COLORS = {
    "Excellent": "#4CAF50",
    "Good":      "#8BC34A",
    "Fair":      "#FFEB3B",
    "Poor":      "#FF9800",
    "Bad":       "#F44336",
}

def activity_score_chart_png(city_label: str, scored_days: list[dict[str, Any]]) -> bytes:
    labels  = [f"{d['day_name'][:3]}\n{d['date'][5:]}" for d in scored_days]
    scores  = [d["score"] for d in scored_days]
    colors  = [_SCORE_COLORS.get(d["verdict"], "#9E9E9E") for d in scored_days]

    fig, ax = _dark_fig(10, 5)
    bars = ax.bar(labels, scores, color=colors, edgecolor=_BORDER, linewidth=0.8, zorder=3)
    for bar, s, d in zip(bars, scores, scored_days):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{d['emoji']} {s}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=_TEXT)

    ax.set_ylim(0, 115)
    ax.set_xlabel("Day", color=_SUBTEXT, fontsize=11)
    ax.set_ylabel("Activity Score (0–100)", color=_SUBTEXT, fontsize=11)
    ax.set_title(f"Outdoor Activity Score — {city_label}",
                 color=_TEXT, fontsize=14, fontweight="bold", pad=12)

    legend_patches = [mpatches.Patch(color=c, label=l) for l, c in _SCORE_COLORS.items()]
    ax.legend(handles=legend_patches, facecolor=_BORDER, edgecolor=_BORDER,
              labelcolor=_TEXT, fontsize=8, loc="upper right")
    return _save(fig)


# ---------------------------------------------------------------------------
# 4. UV + Heatwave chart (single panel, dual y-axis)
# ---------------------------------------------------------------------------
_UV_COLORS = ["#4CAF50", "#FFEB3B", "#FF9800", "#F44336", "#9C27B0"]

def uv_heat_chart_png(city_label: str, uv_heat_days: list[dict[str, Any]]) -> bytes:
    labels = [f"{d['day_name'][:3]}\n{d['date'][5:]}" for d in uv_heat_days]
    uvs    = [d["uv_index"] or 0 for d in uv_heat_days]
    feels  = [d["feels_like_c"] or 20 for d in uv_heat_days]
    x      = np.arange(len(labels))

    def _uv_color(uv: float) -> str:
        if uv >= 11: return _UV_COLORS[4]
        if uv >= 8:  return _UV_COLORS[3]
        if uv >= 6:  return _UV_COLORS[2]
        if uv >= 3:  return _UV_COLORS[1]
        return _UV_COLORS[0]

    fig, ax1 = plt.subplots(figsize=(_W, _H))
    fig.patch.set_facecolor(_BG)
    ax1.set_facecolor(_PANEL)
    ax1.tick_params(axis="both", colors=_SUBTEXT, labelsize=11)
    for spine in ax1.spines.values():
        spine.set_edgecolor(_BORDER)

    # UV bars on left axis
    uv_colors = [_uv_color(u) for u in uvs]
    bars = ax1.bar(x - 0.2, uvs, width=0.4, color=uv_colors,
                   edgecolor=_BORDER, linewidth=0.8, label="UV Index", zorder=3)
    for bar, u in zip(bars, uvs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                 f"{u:.0f}", ha="center", va="bottom", fontsize=10,
                 fontweight="bold", color=_TEXT)
    ax1.set_ylabel("UV Index", color=_SUBTEXT, fontsize=12)
    ax1.set_ylim(0, max(uvs) * 1.4 + 2)
    ax1.grid(axis="y", color=_BORDER, alpha=0.4, zorder=1)

    # Feels-like bars on right axis
    ax2 = ax1.twinx()
    ax2.set_facecolor("none")
    feels_colors = ["#9C27B0" if f >= 38 else "#F44336" if f >= 33 else "#FF9800" if f >= 27 else "#4CAF50" for f in feels]
    bars2 = ax2.bar(x + 0.2, feels, width=0.4, color=feels_colors,
                    alpha=0.75, edgecolor=_BORDER, linewidth=0.8, label="Feels Like (°C)", zorder=3)
    for bar, f in zip(bars2, feels):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{f:.0f}°C", ha="center", va="bottom", fontsize=10,
                 fontweight="bold", color=_TEXT)
    ax2.set_ylabel("Feels Like (°C)", color=_SUBTEXT, fontsize=12)
    ax2.tick_params(colors=_SUBTEXT, labelsize=11)
    ax2.set_ylim(0, max(feels) * 1.4 + 5)
    ax2.spines["right"].set_edgecolor(_BORDER)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, color=_SUBTEXT, fontsize=11)
    ax1.set_title(f"UV Index & Feels-Like Temperature — {city_label}",
                  color=_TEXT, fontsize=14, fontweight="bold", pad=14)

    uv_legend = [mpatches.Patch(color=c, label=l)
                 for c, l in zip(_UV_COLORS, ["UV Low", "UV Moderate", "UV High", "UV Very High", "UV Extreme"])]
    heat_legend = [
        mpatches.Patch(color="#4CAF50", label="Comfortable (<27°C)"),
        mpatches.Patch(color="#FF9800", label="Warm (27–32°C)"),
        mpatches.Patch(color="#F44336", label="Hot (33–37°C)"),
        mpatches.Patch(color="#9C27B0", label="Extreme (≥38°C)"),
    ]
    ax1.legend(handles=uv_legend + heat_legend, facecolor=_BORDER, edgecolor=_BORDER,
               labelcolor=_TEXT, fontsize=9, loc="upper left", ncol=2)

    plt.tight_layout()
    return _save(fig)


# ---------------------------------------------------------------------------
# 5. Air quality chart (single panel, AQI bars + PM lines on right axis)
# ---------------------------------------------------------------------------
_AQI_COLORS = ["#4CAF50", "#8BC34A", "#FFEB3B", "#FF9800", "#F44336", "#9C27B0"]

def air_quality_chart_png(city_label: str, aq_days: list[dict[str, Any]]) -> bytes:
    labels = [d["date"][5:] for d in aq_days]
    pm25   = [d["pm2_5"] or 0 for d in aq_days]
    pm10   = [d["pm10"] or 0 for d in aq_days]
    aqis   = [d["european_aqi"] or 0 for d in aq_days]
    x      = np.arange(len(labels))

    def _aqi_color(aqi: float) -> str:
        if aqi >= 100: return _AQI_COLORS[5]
        if aqi >= 80:  return _AQI_COLORS[4]
        if aqi >= 60:  return _AQI_COLORS[3]
        if aqi >= 40:  return _AQI_COLORS[2]
        if aqi >= 20:  return _AQI_COLORS[1]
        return _AQI_COLORS[0]

    fig, ax1 = plt.subplots(figsize=(_W, _H))
    fig.patch.set_facecolor(_BG)
    ax1.set_facecolor(_PANEL)
    ax1.tick_params(axis="both", colors=_SUBTEXT, labelsize=11)
    for spine in ax1.spines.values():
        spine.set_edgecolor(_BORDER)

    # AQI bars on left axis
    aqi_colors = [_aqi_color(a) for a in aqis]
    bars = ax1.bar(x, aqis, color=aqi_colors, edgecolor=_BORDER, linewidth=0.8,
                   label="European AQI", zorder=3, width=0.5)
    for bar, a, d in zip(bars, aqis, aq_days):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                 d["aqi_category"], ha="center", va="bottom",
                 fontsize=9, fontweight="bold", color=_TEXT)
    ax1.set_ylabel("European AQI", color=_SUBTEXT, fontsize=12)
    ax1.set_ylim(0, max(aqis) * 1.45 + 10)
    ax1.grid(axis="y", color=_BORDER, alpha=0.4, zorder=1)

    # PM2.5 and PM10 lines on right axis
    ax2 = ax1.twinx()
    ax2.set_facecolor("none")
    ax2.plot(x, pm25, color="#E91E63", linewidth=2.5, marker="o",
             markersize=7, label="PM2.5 (μg/m³)", zorder=4)
    ax2.plot(x, pm10, color="#FF9800", linewidth=2.5, marker="s",
             markersize=7, label="PM10 (μg/m³)", zorder=4)
    ax2.axhline(25, color="#E91E63", linestyle="--", alpha=0.5, linewidth=1)
    ax2.axhline(50, color="#FF9800", linestyle="--", alpha=0.5, linewidth=1)
    ax2.set_ylabel("Particulates (μg/m³)", color=_SUBTEXT, fontsize=12)
    ax2.tick_params(colors=_SUBTEXT, labelsize=11)
    ax2.set_ylim(0, max(max(pm25), max(pm10)) * 1.5 + 10)
    ax2.spines["right"].set_edgecolor(_BORDER)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, color=_SUBTEXT, fontsize=11)
    ax1.set_title(f"Air Quality Forecast — {city_label}",
                  color=_TEXT, fontsize=14, fontweight="bold", pad=14)

    aqi_legend = [mpatches.Patch(color=c, label=l) for c, l in zip(
        _AQI_COLORS, ["Good", "Fair", "Moderate", "Poor", "Very Poor", "Extremely Poor"])]
    pm_legend = [
        plt.Line2D([0], [0], color="#E91E63", linewidth=2, marker="o", label="PM2.5 (μg/m³)"),
        plt.Line2D([0], [0], color="#FF9800", linewidth=2, marker="s", label="PM10 (μg/m³)"),
    ]
    ax1.legend(handles=aqi_legend + pm_legend, facecolor=_BORDER, edgecolor=_BORDER,
               labelcolor=_TEXT, fontsize=9, loc="upper left", ncol=2)

    plt.tight_layout()
    return _save(fig)
