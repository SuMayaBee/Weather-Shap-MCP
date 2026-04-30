# weather-shap-mcp

> **Weather intelligence MCP server with real SHAP explainability** — rain, UV, heat & air quality forecasts for any city, powered by XGBoost + `shap.TreeExplainer` + live [Open-Meteo](https://open-meteo.com/) data. No API key required.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

---

## What makes this different

Most weather MCP servers return raw forecast numbers. This one also tells you **why** — using real [SHAP](https://shap.readthedocs.io/) `TreeExplainer` values from trained XGBoost models:

- **4 separate XGBoost models** per city — rain (classifier), UV index (regressor), feels-like temperature (regressor), air quality AQI (regressor)
- **Real `shap.TreeExplainer`** — exact Shapley values, not approximations or hand-written scores
- **SHAP waterfall plots** returned as inline images directly in the chat
- **Trained on 2 years of live historical data** per city — models adapt to local climate patterns
- **Zero API keys** — all data from Open-Meteo's free endpoints

---

## Example conversations

### Rain forecast

**Ask:** `"Will it rain in Dhaka this week?"`

**Returns:** 7-day bar chart with daily rain probabilities

<img width="900" alt="Rain forecast" src="https://github.com/user-attachments/assets/10fa161a-3bdc-4b98-881f-7db81edfd096" />

---

### SHAP explanation — why will it rain?

**Ask:** `"Why is it predicted to rain on day 1?"`

**Returns:** SHAP waterfall plot + plain-English — *"Heavy forecast precipitation (7.9 mm) strongly pushes rain chance up. Warm overnight low (26°C) moderately pushes rain chance up..."*

<img width="900" alt="SHAP rain explanation" src="https://github.com/user-attachments/assets/3368cbef-c637-45a7-954d-55a40e32e68c" />

---

### UV + heat forecast

**Ask:** `"How hot and sunny will it be in Dubai this week?"`

**Returns:** Dual-axis chart — UV index bars + feels-like temperature line

<img width="900" alt="UV and heat forecast" src="https://github.com/user-attachments/assets/1e0e0002-efbb-41f2-b034-71205c87549b" />

---

### Air quality forecast

**Ask:** `"Is the air safe to breathe in Delhi today?"`

**Returns:** AQI bars with PM2.5/PM10 lines and WHO limit markers

<img width="900" alt="Air quality forecast" src="https://github.com/user-attachments/assets/4437a4f8-6d2e-4ade-bc4f-dce73fdf4076" />

---

### SHAP explanation — why is air quality poor?

**Ask:** `"Why is the air quality so poor on day 1?"`

**Returns:** SHAP waterfall — *"Very low wind (3 km/h — pollutants accumulate) strongly worsens air quality. No rain (0 mm — no washout) moderately worsens air quality..."*

<img width="900" alt="SHAP air quality explanation" src="https://github.com/user-attachments/assets/5014ccc4-2d02-413e-b415-c69a22d99050" />

---

### Best day finder

**Ask:** `"What's the best day for an outdoor event in Tokyo this week?"`

**Returns:** *"Thursday (score 82/100 — Excellent). Low rain chance (12%), comfortable temperature (24°C), UV moderate."*

<img width="900" alt="Best day finder" src="https://github.com/user-attachments/assets/175a0194-aa6f-4d84-b7ad-ee0fdb305708" />

---

## Tools

| Tool | What it does | Returns |
|---|---|---|
| `predict_rain_7day` | Rain probability forecast for next N days | Bar chart + JSON |
| `explain_rain_prediction` | SHAP waterfall for rain prediction on a specific day | Waterfall plot + plain-English reasons |
| `get_uv_and_heat_forecast` | UV index + feels-like temperature forecast | Dual-axis chart + JSON |
| `explain_uv_prediction` | SHAP waterfall explaining UV intensity drivers | Waterfall plot + reasons |
| `explain_heat_prediction` | SHAP waterfall explaining why it feels hot/cool | Waterfall plot + reasons |
| `get_air_quality` | PM2.5, PM10, European AQI forecast | AQI chart + JSON |
| `explain_air_quality_prediction` | SHAP waterfall explaining pollution drivers | Waterfall plot + reasons |
| `get_outdoor_activity_score` | 0–100 daily score for outdoor activities | Colour-coded bar chart + reasons |
| `find_best_day_for` | Best day this week for running / beach / picnic / event | Ranked answer |
| `city_model_info` | Trained model accuracy + training window | JSON |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/SuMayaBee/weather-shap-mcp.git
cd weather-shap-mcp
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv

# Activate (Linux / macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

pip install -e .
```

### 3. Verify it works

```bash
venv/bin/weather-mcp
# Server starts and waits for MCP connections over stdio — Ctrl+C to stop
```

On Windows: `venv\Scripts\weather-mcp.exe`

---

## Connect to Cursor

The repo includes a `.cursor/mcp.json` — open the cloned folder directly in Cursor and it picks it up automatically.

Or add manually in **Cursor Settings → Tools & MCP → Add new MCP server**:

```json
{
  "mcpServers": {
    "weather": {
      "command": "/absolute/path/to/weather-shap-mcp/venv/bin/weather-mcp"
    }
  }
}
```

Replace `/absolute/path/to/` with the actual path where you cloned the repo.

## Connect to VS Code

The repo includes a `.vscode/mcp.json`. Or add to your workspace `.vscode/mcp.json`:

```json
{
  "servers": {
    "weather": {
      "type": "stdio",
      "command": "/absolute/path/to/weather-shap-mcp/venv/bin/weather-mcp"
    }
  }
}
```

## Connect to Claude Desktop

Edit `~/.config/Claude/claude_desktop_config.json` (Linux) or
`~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "weather": {
      "command": "/absolute/path/to/weather-shap-mcp/venv/bin/weather-mcp"
    }
  }
}
```

---

## How it works

```
User query
    │
    ▼
Geocode city (Open-Meteo geocoding API)
    │
    ▼
Fetch 2 years of daily archive data (Open-Meteo archive API)
    │   precipitation, temp_max, temp_min, windspeed,
    │   uv_index_max, apparent_temperature_max
    │
    ▼
Train 4 XGBoost models per city
    ├── Rain classifier      → predict_proba (rain tomorrow?)
    ├── UV regressor         → predict UV index
    ├── Heat regressor       → predict feels-like temperature
    └── AQI regressor        → predict air quality (90-day AQ history)
    │
    ▼
Fetch live 7-day forecast (Open-Meteo forecast API)
    │
    ▼
Run inference + shap.TreeExplainer on all 4 models
    │
    ▼
Return JSON + matplotlib chart as inline PNG image
```

**SHAP implementation:**
- `shap.TreeExplainer` — exact Shapley values (not sampling-based approximation)
- `shap.Explanation` + `shap.plots.waterfall` — standard SHAP waterfall plots
- Same 4 features across all models: `[precipitation, temp_max, temp_min, windspeed]`
- Each model has its own explainer — SHAP contributions are model-specific, so the same feature can push rain up but push AQI down

---

## Project structure

```
weather_mcp/
├── server.py       # FastMCP server — 10 tools
├── model.py        # XGBoost training, SHAP explain functions, plain-English generators
├── open_meteo.py   # Open-Meteo API client (geocode, archive, forecast, air quality)
├── scoring.py      # Rule-based activity score, UV/heat/AQI bands
└── visualize.py    # matplotlib chart generators (forecast, SHAP waterfall, AQI, UV)
```

---

## Data sources

| Source | Endpoint | Used for |
|---|---|---|
| Open-Meteo Geocoding | `geocoding-api.open-meteo.com` | City → lat/lon |
| Open-Meteo Archive (ERA5) | `archive-api.open-meteo.com` | 2yr historical training data |
| Open-Meteo Forecast | `api.open-meteo.com/v1/forecast` | Live 7-day inference data |
| Open-Meteo Air Quality (CAMS) | `air-quality-api.open-meteo.com` | PM2.5, PM10, AQI |

No API keys required for any of these.

---

## Notes

- Models are cached in-process per city. Restarting the server retrains on next query (~10–20s for first call per city).
- Location keys are rounded to 2 decimal places (~1 km), so nearby queries share the same cached model.
- If archive UV data is unavailable for a location (ERA5 coverage gap), the UV model is trained on synthesised data derived from temperature + precipitation correlations.
- AQI model uses 90 days of historical air quality data (Open-Meteo CAMS limit). Requires ≥ 30 days to train — falls back gracefully if unavailable.

---

## License

MIT
