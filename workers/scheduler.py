"""
workers/scheduler.py

Background worker that polls CWC / WB-SWD telemetry endpoints hourly for
every registered station and dataset, writes results to data/processed/,
and evaluates district thresholds to raise warning/danger flags.

Run standalone:   python -m workers.scheduler
Or via Celery beat (see docker-compose.yml `worker` service) for
production process supervision/retries.
"""

import json
import time
from datetime import datetime, timezone

from config.settings import (
    DATASETS, TELEMETRY_POLL_INTERVAL_MIN, PROCESSED_DATA_DIR, THRESHOLDS_PATH,
)
from utils.api_client import fetch_telemetry
from utils.registry import STATIONS
from utils.logger import get_logger

logger = get_logger(__name__)


def load_thresholds() -> dict:
    with open(THRESHOLDS_PATH) as f:
        return json.load(f)


def evaluate_alert(district: str, level_m: float, thresholds: dict) -> str:
    cfg = thresholds["districts"].get(district)
    if cfg is None:
        return "unknown"
    if level_m >= cfg["extreme_danger_level_m"]:
        return "extreme_danger"
    if level_m >= cfg["danger_level_m"]:
        return "danger"
    if level_m >= cfg["warning_level_m"]:
        return "warning"
    return "normal"


def poll_once() -> list[dict]:
    """Poll every station/dataset once; return list of alert records."""
    thresholds = load_thresholds()
    results = []

    for station in STATIONS.values():
        for dataset_key in ("river_water_level", "rainfall"):
            payload = fetch_telemetry(station.station_id, dataset_key, hours_back=6)
            if not payload:
                continue

            latest_level = payload.get("latest_value")
            if latest_level is None or dataset_key != "river_water_level":
                continue

            status = evaluate_alert(station.district, latest_level, thresholds)
            record = {
                "station_id": station.station_id,
                "district": station.district,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level_m": latest_level,
                "status": status,
            }
            results.append(record)
            if status in ("danger", "extreme_danger"):
                logger.warning("threshold breach", extra=record)

    out_path = PROCESSED_DATA_DIR / f"alerts_{int(time.time())}.json"
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_forever():
    logger.info("scheduler starting", extra={"interval_min": TELEMETRY_POLL_INTERVAL_MIN})
    while True:
        try:
            poll_once()
        except Exception as exc:  # noqa: BLE001 -- keep the loop alive
            logger.error("poll cycle failed", extra={"error": str(exc)})
        time.sleep(TELEMETRY_POLL_INTERVAL_MIN * 60)


if __name__ == "__main__":
    run_forever()
