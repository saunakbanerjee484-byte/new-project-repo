"""
models/trainer.py

Training/validation for the 6h and 12h water-level forecast models.
XGBoost is the primary/production model (fast, robust on tabular
lag-features, easy to retrain incrementally).
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import xgboost as xgb

from config.settings import FORECAST_HORIZONS_HOURS, PROCESSED_DATA_DIR
from models.features import build_feature_matrix
from utils.logger import get_logger

logger = get_logger(__name__)

MODEL_DIR = PROCESSED_DATA_DIR.parent / "models_store"


def train_xgboost(df: pd.DataFrame, target_col: str = "water_level_m",
                  horizon_hours: int = 6, n_splits: int = 5):
    """
    Trains an XGBoost regression model using Auto-Regressive Distributed Lag (ARDL) features.
    """
    feat_df = build_feature_matrix(df, target_col)
    target_name = f"target_{target_col}_{horizon_hours}h"
    feature_cols = [c for c in feat_df.columns if not c.startswith("target_")]

    X, y = feat_df[feature_cols], feat_df[target_name]
    tscv = TimeSeriesSplit(n_splits=n_splits)

    scores = []
    model = None
    
    logger.info(f"Starting XGBoost Training for {horizon_hours}h horizon...")
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        model = xgb.XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[val_idx])
        
        mae = mean_absolute_error(y.iloc[val_idx], pred)
        rmse = root_mean_squared_error(y.iloc[val_idx], pred)
        scores.append({"fold": fold, "mae": mae, "rmse": rmse})
        logger.info(f"Fold {fold} trained -> MAE: {mae:.3f}, RMSE: {rmse:.3f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"xgb_{target_col}_{horizon_hours}h.joblib"
    joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)
    
    logger.info(f"Model saved successfully at: {model_path}")

    return {"model_path": str(model_path), "cv_scores": scores}