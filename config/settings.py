"""
config/settings.py

Centralized environment configuration for DeltaPulse. All values are
read from environment variables (with sane local-dev defaults) so the
same codebase runs unmodified in Docker/production.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --- API endpoints (CWC / West Bengal Surface Water Department) ------------
CWC_API_BASE_URL = os.getenv("CWC_API_BASE_URL", "https://indiawris.gov.in/api")
CWC_API_KEY = os.getenv("CWC_API_KEY", "")
API_REQUEST_TIMEOUT_S = int(os.getenv("API_REQUEST_TIMEOUT_S", "15"))
API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "3"))
API_RATE_LIMIT_PER_MIN = int(os.getenv("API_RATE_LIMIT_PER_MIN", "60"))

DATASETS = {
    "river_water_level": "River Water Level (Telemetry - Hourly), West Bengal Surface Water Department",
    "rainfall": "Rainfall (Telemetry - Hourly), West Bengal-Surface Water (Bengal-SW)",
    "durgapur_barrage_discharge": "Reservoir Discharge, Durgapur Barrage_2, West Bengal (Telemetry - Hourly)",
    "reservoir_level": "Reservoir Water Level (Manual-Daily), West Bengal Surface Water Department",
    "panchet_dam_discharge": "Reservoir Discharge, Durgapur Scada-Panchet Dam, West Bengal (Telemetry - Hourly)",
}

# --- Database ----------------------------------------------------------------
POSTGRES_DSN = os.getenv(
    "POSTGRES_DSN", "postgresql://deltapulse:deltapulse@localhost:5432/deltapulse"
)
USE_TIMESCALEDB = os.getenv("USE_TIMESCALEDB", "true").lower() == "true"

# --- Cache (Redis / in-memory fallback) --------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "900"))  # 15 min, matches hourly telemetry cadence
USE_REDIS = os.getenv("USE_REDIS", "true").lower() == "true"

# --- Scheduler ----------------------------------------------------------------
TELEMETRY_POLL_INTERVAL_MIN = int(os.getenv("TELEMETRY_POLL_INTERVAL_MIN", "60"))

# --- Paths ---------------------------------------------------------------------
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
THRESHOLDS_PATH = BASE_DIR / "config" / "thresholds.json"

# --- Forecasting ---------------------------------------------------------------
FORECAST_HORIZONS_HOURS = [6, 12]
LAG_FEATURES_HOURS = [1, 2, 3, 6, 12, 24]

# --- Logging ---------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "json")  # "json" or "console"
