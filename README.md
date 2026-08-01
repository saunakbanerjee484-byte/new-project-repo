# DeltaPulse (FlowState) -- West Bengal Flood Intelligence Platform

Production-grade real-time flood intelligence and river analytics platform
combining classical open-channel mechanics, closed-conduit pipe networks,
numerical simulation, urban drainage modeling, and machine learning.

## Status

| Module | Status |
|---|---|
| `engines/dam_break.py` -- 1D/2D Saint-Venant dam-break & embankment breach | ✅ Implemented, unit-tested against Ritter (1892) analytical solution |
| `engines/rating_curve.py` -- Q = a(h-h0)^b calibration | ✅ Implemented |
| `engines/routing.py` -- Muskingum-Cunge routing | ✅ Implemented |
| `engines/sediment.py` -- Shields parameter | ✅ Implemented |
| `engines/bridge_torrents.py` -- supercritical flow & HEC-18 pier scour | ✅ Implemented |
| `engines/embankment.py` -- Casagrande phreatic line & piping FoS | ✅ Implemented |
| `models/*.py` -- lag features, XGBoost/LSTM trainer, forecaster, scour ML | 🟡 Scaffolded (needs a real training dataset) |
| `geospatial/swmm_runner.py` -- EPA SWMM wrapper | 🟡 Scaffolded (needs a calibrated `.inp` network model) |
| `geospatial/inundation_map.py` | 🟡 Scaffolded |
| `utils/*.py` -- api_client, cache, registry, logger | ✅ Implemented |
| `workers/scheduler.py` | ✅ Implemented (point it at a real CWC/WB-SWD endpoint + API key) |
| `dam_break_dashboard.py` / `app.py` | ✅ Interactive Streamlit dam-break lab implemented |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run the interactive dam-break simulator standalone
streamlit run dam_break_dashboard.py

# or the full command center (dam-break tab + status/hydrograph/forecast stubs)
streamlit run app.py

# run tests (dam-break verified against Ritter's analytical solution)
PYTHONPATH=. pytest tests/ -q
```

## Full stack (Docker)

```bash
docker compose up --build
```
Starts the Streamlit app, a Celery-style scheduler worker, TimescaleDB, and Redis.

## Architecture

```
wb_flood_intelligence/
├── config/        Environment config & district thresholds
├── data/           raw/ + processed/ telemetry
├── utils/          API client, cache, registry, logging
├── engines/        Hydraulic / geotechnical / numerical engines
├── models/         ML forecasting (features, trainer, forecaster, scour ML)
├── geospatial/      SWMM wrapper + inundation mapping
├── workers/        Hourly telemetry polling & threshold alerting
├── tests/          pytest suite (incl. Ritter analytical verification)
├── app.py           Streamlit Command Center
├── dam_break_dashboard.py   Standalone dam-break/breach simulator UI
├── docker-compose.yml, Dockerfile, requirements.txt
```

## Dam-Break / Embankment Breach Engine (`engines/dam_break.py`)

Two linked workflows:

1. **Breach hydrograph generation** -- Froehlich (1995/2008) empirical
   breach geometry, discharged through a linearly-growing trapezoidal
   breach via the broad-crested weir equation, with optional lumped
   reservoir drawdown.
2. **Flood-wave routing** -- HLL (Harten-Lax-van Leer) finite-volume
   solution of the 1D Saint-Venant equations, robust across the
   wet/dry front and the subcritical<->supercritical transition a
   dam-break wave always produces. A 2D variant (Strang-split HLL) is
   included for floodplain/urban inundation-extent mapping.

Verified in `tests/test_dam_break.py` against Ritter's (1892) closed-form
dry-bed dam-break solution, plus a still-water well-balancedness check
and a short-time mass-conservation check.

## Next steps

- Point `utils/api_client.py` at the live CWC/WB-SWD telemetry endpoints
  and set `CWC_API_KEY`.
- Populate `config/thresholds.json` with real gauge-datum danger/warning
  levels per district.
- Train `models/trainer.py` on historical telemetry once
  `data/processed/` has enough hourly history (needs several months for
  the 6h/12h XGBoost models to be meaningful).
- Build/calibrate a real SWMM `.inp` network from the district
  storm-drain GIS layer for `geospatial/swmm_runner.py`.
- Advanced-features backlog: Digital Twin/BIM linkage, radar/satellite
  assimilation for 24-48h lead times, automated SMS/Telegram dispatch
  to district disaster-management authorities.
