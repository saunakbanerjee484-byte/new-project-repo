"""
models/forecaster.py

Inference-time wrapper: loads a trained model (models/trainer.py output)
and produces 6h/12h water-level point forecasts (plus a simple residual-
based uncertainty band) from the latest telemetry window.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from config.settings import PROCESSED_DATA_DIR, FORECAST_HORIZONS_HOURS
from models.features import build_feature_matrix
from utils.logger import get_logger

logger = get_logger(__name__)

MODEL_DIR = PROCESSED_DATA_DIR.parent / "models_store"


class WaterLevelForecaster:
    def __init__(self, target_col: str = "water_level_m"):
        self.target_col = target_col
        self._models = {}

    def _load(self, horizon_hours: int):
        if horizon_hours in self._models:
            return self._models[horizon_hours]
        path = MODEL_DIR / f"xgb_{self.target_col}_{horizon_hours}h.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"No trained model at {path} -- run models.trainer.train_xgboost first."
            )
        bundle = joblib.load(path)
        self._models[horizon_hours] = bundle
        return bundle

    def predict(self, recent_df: pd.DataFrame, horizon_hours: int = 6) -> dict:
        """
        recent_df must contain at least (lag window + rolling window)
        hours of history ending at 'now', with the same raw columns used
        at training time.
        """
        bundle = self._load(horizon_hours)
        model, feature_cols = bundle["model"], bundle["feature_cols"]

        feat_df = build_feature_matrix(recent_df, self.target_col)
        if feat_df.empty:
            raise ValueError("insufficient history to build features for prediction")

        latest = feat_df[feature_cols].iloc[[-1]]
        point_forecast = float(model.predict(latest)[0])

        # crude uncertainty band from in-sample residual std (replace with
        # conformal prediction / quantile model for production-grade bands)
        preds_in_sample = model.predict(feat_df[feature_cols])
        target_name = f"target_{self.target_col}_{horizon_hours}h"
        resid_std = float(np.std(feat_df[target_name] - preds_in_sample))

        return {
            "horizon_hours": horizon_hours,
            "forecast_m": point_forecast,
            "lower_90_m": point_forecast - 1.645 * resid_std,
            "upper_90_m": point_forecast + 1.645 * resid_std,
        }

    def predict_all_horizons(self, recent_df: pd.DataFrame) -> list[dict]:
        return [self.predict(recent_df, h) for h in FORECAST_HORIZONS_HOURS]
