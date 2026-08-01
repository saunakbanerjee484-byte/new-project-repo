"""
utils/api_client.py

Robust fetchers for CWC / West Bengal Surface Water Department telemetry
APIs: retry with exponential backoff, basic rate limiting, and response
caching via utils/cache.py.
"""

import time
import logging
from typing import Optional

import requests
from requests.adapters import HTTPAdapter, Retry

from config.settings import (
    CWC_API_BASE_URL, CWC_API_KEY, API_REQUEST_TIMEOUT_S,
    API_MAX_RETRIES, API_RATE_LIMIT_PER_MIN,
)
from utils.cache import get_cached, set_cached
from utils.logger import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """Simple token-bucket rate limiter, shared per-process."""

    def __init__(self, calls_per_min: int):
        self.min_interval = 60.0 / max(calls_per_min, 1)
        self._last_call = 0.0

    def wait(self):
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()


_rate_limiter = RateLimiter(API_RATE_LIMIT_PER_MIN)


def _build_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=API_MAX_RETRIES,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.mount("http://", HTTPAdapter(max_retries=retries))
    return session


_session = _build_session()


def fetch_telemetry(station_id: str, dataset_key: str, hours_back: int = 24,
                     use_cache: bool = True) -> Optional[dict]:
    """
    Fetch hourly telemetry (water level / rainfall / discharge) for a
    station. Returns parsed JSON, or None on unrecoverable failure.
    """
    cache_key = f"telemetry:{dataset_key}:{station_id}:{hours_back}"
    if use_cache:
        cached = get_cached(cache_key)
        if cached is not None:
            logger.debug("cache hit", extra={"key": cache_key})
            return cached

    _rate_limiter.wait()
    params = {"stationId": station_id, "dataset": dataset_key, "hours": hours_back}
    headers = {"Authorization": f"Bearer {CWC_API_KEY}"} if CWC_API_KEY else {}

    try:
        resp = _session.get(
            f"{CWC_API_BASE_URL}/telemetry", params=params, headers=headers,
            timeout=API_REQUEST_TIMEOUT_S,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        logger.error("telemetry fetch failed", extra={"station_id": station_id, "error": str(exc)})
        return None

    if use_cache:
        set_cached(cache_key, data)
    return data
