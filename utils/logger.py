"""
utils/logger.py

Structured logging: JSON logs in production (easy to ship to a log
aggregator), readable console logs in local dev.
"""

import json
import logging
import sys

from config.settings import LOG_LEVEL, LOG_FORMAT


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # include any structured `extra=` fields
        reserved = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())
        for key, val in record.__dict__.items():
            if key not in reserved:
                payload[key] = val
        return json.dumps(payload, default=str)


_configured_loggers: set[str] = set()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if name in _configured_loggers:
        return logger

    logger.setLevel(LOG_LEVEL)
    handler = logging.StreamHandler(sys.stdout)
    if LOG_FORMAT == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        ))
    logger.addHandler(handler)
    logger.propagate = False
    _configured_loggers.add(name)
    return logger
