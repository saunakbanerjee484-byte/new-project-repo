"""
utils/cache.py

Caching layer for live API responses. Uses Redis when USE_REDIS=true and
reachable; otherwise transparently falls back to an in-process TTL
dict cache so the app still runs without a Redis instance (e.g. local dev).
"""

import json
import time
from typing import Any, Optional

from config.settings import REDIS_URL, CACHE_TTL_SECONDS, USE_REDIS
from utils.logger import get_logger

logger = get_logger(__name__)

_memory_cache: dict[str, tuple[float, Any]] = {}

_redis_client = None
if USE_REDIS:
    try:
        import redis
        _redis_client = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=2)
        _redis_client.ping()
        logger.info("connected to redis", extra={"url": REDIS_URL})
    except Exception as exc:  # noqa: BLE001 -- any connectivity issue -> fallback
        logger.warning("redis unavailable, falling back to in-memory cache", extra={"error": str(exc)})
        _redis_client = None


def get_cached(key: str) -> Optional[Any]:
    if _redis_client is not None:
        raw = _redis_client.get(key)
        return json.loads(raw) if raw else None

    entry = _memory_cache.get(key)
    if entry is None:
        return None
    expires_at, value = entry
    if time.time() > expires_at:
        _memory_cache.pop(key, None)
        return None
    return value


def set_cached(key: str, value: Any, ttl: int = CACHE_TTL_SECONDS) -> None:
    if _redis_client is not None:
        _redis_client.setex(key, ttl, json.dumps(value))
        return
    _memory_cache[key] = (time.time() + ttl, value)


def invalidate(key: str) -> None:
    if _redis_client is not None:
        _redis_client.delete(key)
    _memory_cache.pop(key, None)
