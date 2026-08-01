"""
models/trainer.py

Training/validation for the 6h and 12h water-level forecast models.
XGBoost is the primary/production model (fast, robust on tabular
lag-features, easy to retrain incrementally); an LSTM variant is
provided for comparison on longer sequential dependencies.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from config.settings import FORECAST_HORIZONS_HOURS, PROCESSED_DATA_DIR
from models.features import build_feature_matrix
from utils.logger import get_logger

logger = get_logger(__name__)

MODEL_DIR = PROCESSED_DATA_DIR.parent / "models_store"


def train_xgboost(df: pd.DataFrame, target_col: str = "water_level_m",
                   horizon_hours: int = 6, n_splits: int = 5):
    import xgboost as xgb

    feat_df = build_feature_matrix(df, target_col)
    target_name = f"target_{target_col}_{horizon_hours}h"
    feature_cols = [c for c in feat_df.columns if not c.startswith("target_")]

    X, y = feat_df[feature_cols], feat_df[target_name]
    tscv = TimeSeriesSplit(n_splits=n_splits)

    scores = []
    model = None
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
        logger.info("xgb fold trained", extra=scores[-1])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"xgb_{target_col}_{horizon_hours}h.joblib"
    joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)

    return {"model_path": str(model_path), "cv_scores": scores}


def train_lstm(df: pd.DataFrame, target_col: str = "water_level_m",
                horizon_hours: int = 6, sequence_len: int = 24, epochs: int = 20):
    """
    LSTM baseline for comparison against XGBoost on longer temporal
    dependencies. Kept separate/optional so the core pipeline doesn't
    hard-require a deep-learning framework in lightweight deployments.
    """
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler

    values = df[target_col].values.reshape(-1, 1)
    scaler = StandardScaler().fit(values)
    scaled = scaler.transform(values).flatten()

    X, y = [], []
    for i in range(len(scaled) - sequence_len - horizon_hours):
        X.append(scaled[i:i + sequence_len])
        y.append(scaled[i + sequence_len + horizon_hours - 1])
    X, y = np.array(X)[..., None], np.array(y)

    split = int(len(X) * 0.85)
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=(sequence_len, 1)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    history = model.fit(X[:split], y[:split], validation_data=(X[split:], y[split:]),
                         epochs=epochs, batch_size=32, verbose=0)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"lstm_{target_col}_{horizon_hours}h.keras"
    model.save(model_path)
    joblib.dump(scaler, MODEL_DIR / f"lstm_{target_col}_{horizon_hours}h_scaler.joblib")

    return {"model_path": str(model_path), "final_val_loss": float(history.history["val_loss"][-1])}
