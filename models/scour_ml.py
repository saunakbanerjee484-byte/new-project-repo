"""
models/scour_ml.py

Prediction of Local Scour Around Bridge Piers: Empirical vs. Machine
Learning Models.

Benchmarks the classical HEC-18/CSU empirical formula
(engines.bridge_torrents.hec18_pier_scour_depth) against a gradient-
boosted regression trained on flume/field scour datasets, since HEC-18
is known to over-predict for many field conditions -- an ML model
trained on the same governing dimensionless groups (Fr1, y1/a, d50/a,
sigma_g) typically tightens that bias.
"""

from dataclasses import dataclass
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score

from engines.bridge_torrents import hec18_pier_scour_depth, froude_number
from config.settings import PROCESSED_DATA_DIR
from utils.logger import get_logger

logger = get_logger(__name__)
MODEL_PATH = PROCESSED_DATA_DIR.parent / "models_store" / "scour_gbr.joblib"

FEATURE_COLUMNS = ["froude_number", "y1_over_a", "d50_over_a", "pier_shape_factor"]


@dataclass
class ScourDataset:
    """
    Expected columns: y1 (m), V1 (m/s), pier_width (m), d50 (m),
    pier_shape_factor (1.0 round, 1.1 square), observed_scour_depth (m).
    """
    df: pd.DataFrame

    def to_features(self) -> pd.DataFrame:
        out = self.df.copy()
        out["froude_number"] = froude_number(out["V1"], out["y1"])
        out["y1_over_a"] = out["y1"] / out["pier_width"]
        out["d50_over_a"] = out["d50"] / out["pier_width"]
        return out


def empirical_predictions(dataset: ScourDataset) -> np.ndarray:
    df = dataset.df
    return np.array([
        hec18_pier_scour_depth(row.y1, row.V1, row.pier_width, K1=row.pier_shape_factor)
        for row in df.itertuples()
    ])


def train_scour_model(dataset: ScourDataset, n_splits: int = 5) -> dict:
    feat_df = dataset.to_features()
    X, y = feat_df[FEATURE_COLUMNS], feat_df["observed_scour_depth"]

    model = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_pred = cross_val_predict(model, X, y, cv=cv)

    model.fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    empirical_pred = empirical_predictions(dataset)

    comparison = {
        "ml_mae": float(mean_absolute_error(y, cv_pred)),
        "ml_r2": float(r2_score(y, cv_pred)),
        "empirical_hec18_mae": float(mean_absolute_error(y, empirical_pred)),
        "empirical_hec18_r2": float(r2_score(y, empirical_pred)),
        "model_path": str(MODEL_PATH),
    }
    logger.info("scour model comparison", extra=comparison)
    return comparison


def predict_scour_ml(y1, V1, pier_width, d50, pier_shape_factor=1.0) -> float:
    model = joblib.load(MODEL_PATH)
    fr = froude_number(V1, y1)
    X = pd.DataFrame([{
        "froude_number": fr,
        "y1_over_a": y1 / pier_width,
        "d50_over_a": d50 / pier_width,
        "pier_shape_factor": pier_shape_factor,
    }])
    return float(model.predict(X)[0])
