"""
models/features.py

Lag-feature and rolling-window generators for the water-level forecasting
pipeline (models/trainer.py, models/forecaster.py). Operates on a
pandas DataFrame indexed by hourly timestamp with columns like
'water_level_m', 'rainfall_mm', 'upstream_discharge_m3s'.
"""

import pandas as pd

from config.settings import LAG_FEATURES_HOURS, FORECAST_HORIZONS_HOURS


def add_lag_features(df: pd.DataFrame, columns: list[str],
                      lags_hours: list[int] = None) -> pd.DataFrame:
    lags_hours = lags_hours or LAG_FEATURES_HOURS
    out = df.copy()
    for col in columns:
        for lag in lags_hours:
            out[f"{col}_lag{lag}h"] = out[col].shift(lag)
    return out


def add_rolling_features(df: pd.DataFrame, columns: list[str],
                          windows_hours: list[int] = (3, 6, 12, 24)) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        for w in windows_hours:
            out[f"{col}_rollmean{w}h"] = out[col].rolling(w).mean()
            out[f"{col}_rollmax{w}h"] = out[col].rolling(w).max()
            out[f"{col}_rollstd{w}h"] = out[col].rolling(w).std()
    return out


def add_rate_of_change(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[f"{col}_roc_1h"] = out[col].diff(1)
        out[f"{col}_roc_3h"] = out[col].diff(3)
    return out


def add_targets(df: pd.DataFrame, target_col: str,
                 horizons_hours: list[int] = None) -> pd.DataFrame:
    """Create future-shifted target columns for each forecast horizon."""
    horizons_hours = horizons_hours or FORECAST_HORIZONS_HOURS
    out = df.copy()
    for h in horizons_hours:
        out[f"target_{target_col}_{h}h"] = out[target_col].shift(-h)
    return out


def build_feature_matrix(df: pd.DataFrame, target_col: str = "water_level_m") -> pd.DataFrame:
    """Full feature-engineering pipeline used by both trainer.py and forecaster.py."""
    feature_cols = [c for c in df.columns if not c.startswith("target_")]
    out = add_lag_features(df, feature_cols)
    out = add_rolling_features(out, feature_cols)
    out = add_rate_of_change(out, feature_cols)
    out = add_targets(out, target_col)
    return out.dropna()
