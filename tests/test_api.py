"""
tests/test_api.py

Mock tests for CWC/WB-SWD telemetry endpoint interaction -- no live
network calls, uses monkeypatched requests responses.
"""

from unittest.mock import patch, MagicMock

from utils.api_client import fetch_telemetry
from workers.scheduler import evaluate_alert


def test_fetch_telemetry_returns_parsed_json():
    fake_response = MagicMock()
    fake_response.json.return_value = {"latest_value": 12.3, "unit": "m"}
    fake_response.raise_for_status = lambda: None

    with patch("utils.api_client._session.get", return_value=fake_response):
        result = fetch_telemetry("DURGAPUR_BARRAGE", "river_water_level", use_cache=False)

    assert result == {"latest_value": 12.3, "unit": "m"}


def test_fetch_telemetry_handles_request_failure_gracefully():
    import requests
    with patch("utils.api_client._session.get", side_effect=requests.ConnectionError):
        result = fetch_telemetry("DURGAPUR_BARRAGE", "river_water_level", use_cache=False)
    assert result is None


def test_evaluate_alert_thresholds():
    thresholds = {
        "districts": {
            "Bardhaman": {
                "warning_level_m": 55.0,
                "danger_level_m": 57.5,
                "extreme_danger_level_m": 59.0,
            }
        }
    }
    assert evaluate_alert("Bardhaman", 50.0, thresholds) == "normal"
    assert evaluate_alert("Bardhaman", 56.0, thresholds) == "warning"
    assert evaluate_alert("Bardhaman", 58.0, thresholds) == "danger"
    assert evaluate_alert("Bardhaman", 60.0, thresholds) == "extreme_danger"
