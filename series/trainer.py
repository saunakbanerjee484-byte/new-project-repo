"""
models/trainer.py

Training/Validation for 6h and 12h Water-Level Forecast Models: XGBoost.
=======================================================================
TECHNICAL BACKGROUND (read before using):
This module is XGBoost-only by design, not by omission: an LSTM path
was considered (and is common in the hydrology-ML literature as a
sequence-learning alternative), but was dropped because it needs
TensorFlow, which (a) has no official Python 3.14 wheel as of this
writing and (b) is primarily worth the added complexity when a GPU is
available to train on -- neither holds for this deployment target.
XGBoost (Chen & Guestrin, 2016) is a gradient-boosted tree ensemble
trained on the engineered lag/rolling-window features from
models/features.py; it trains efficiently on CPU alone, handles the
tabular, nonlinear lag-feature relationships common in river forecasting
well, and needs meaningfully less data than a recurrent network to
generalize. Validation below uses TimeSeriesSplit (walk-forward-style,
chronology-respecting folds), which multiple 2025-2026
streamflow-forecasting studies (cited inline below) confirm is standard
practice -- a random k-fold split would let future timesteps leak into
training and produce optimistic, unrealistic validation scores.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from config.settings import settings
from models.features import build_feature_matrix
from utils.logger import get_logger

logger = get_logger(__name__)

MODEL_DIR = settings.MODELS_DIR


def train_xgboost(df: pd.DataFrame, target_col: str = "water_level_m",
                   horizon_hours: int = 6, n_splits: int = 5):
    """
    Trains an XGBoost regressor to predict water level `horizon_hours`
    ahead from the lag/rolling/rate-of-change features in
    models.features.build_feature_matrix().

    THEORY -- why TimeSeriesSplit, not ordinary KFold: standard k-fold
    cross-validation shuffles rows randomly across folds, which for a
    time series means some training rows can sit *chronologically after*
    the validation rows they're being scored against -- the model would
    effectively be allowed to "see the future" via autocorrelated
    neighbors of the leaked-forward rows, inflating the validation score
    beyond what real deployment (where only the past is ever available)
    could achieve. TimeSeriesSplit instead produces `n_splits` folds
    with strictly expanding training windows, each validated only on the
    chronologically-next block -- confirmed as the standard approach for
    hydrological ML forecasting during the web search performed for this
    module (e.g. a 2026 Scientific Reports streamflow-prediction study:
    "TimeSeriesSplit cross-validation with 5 folds was employed instead
    of standard k-fold cross-validation ... training data always
    precedes validation data chronologically").
    """
    import xgboost as xgb

    # build_feature_matrix() (models/features.py) generates the lag,
    # rolling-window, and rate-of-change features, plus the horizon-
    # shifted target columns, dropping rows that don't have a full lag
    # window or a known future target (NaNs at the start/end of the
    # series after shifting).
    feat_df = build_feature_matrix(df, target_col)

    # The specific target column for THIS horizon -- build_feature_matrix
    # creates one such column per horizon in settings.FORECAST_HORIZONS_HOURS,
    # e.g. "target_water_level_m_6h" for a 6-hour-ahead forecast.
    target_name = f"target_{target_col}_{horizon_hours}h"

    # Every column that isn't itself a target is a candidate feature --
    # this naturally includes the horizon-6h run's OWN lag/rolling
    # features even when training the 12h model, keeping the two models'
    # feature sets identical and simplifying downstream inference code.
    feature_cols = [c for c in feat_df.columns if not c.startswith("target_")]

    X, y = feat_df[feature_cols], feat_df[target_name]

    # n_splits folds of strictly chronological train/validation splits,
    # each validation block immediately following (never overlapping)
    # its training block -- see the TimeSeriesSplit rationale above.
    tscv = TimeSeriesSplit(n_splits=n_splits)

    scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        # A fresh model per fold (rather than continuing to train one
        # model across folds) so each fold's score reflects training
        # from scratch on that fold's own expanding window -- this is
        # what makes the fold scores comparable to each other and to a
        # genuine walk-forward deployment scenario.
        fold_model = xgb.XGBRegressor(
            n_estimators=400,      # number of boosting rounds (trees)
            max_depth=6,           # per-tree depth -- controls how much
                                   # feature interaction each tree can
                                   # capture before the ensemble as a
                                   # whole builds up complexity
            learning_rate=0.03,    # shrinkage per boosting round -- a
                                   # small value paired with a large
                                   # n_estimators is the standard
                                   # gradient-boosting regularization
                                   # recipe (Friedman, 2001) to avoid
                                   # overfitting a noisy hydrological
                                   # target
            subsample=0.8,         # row subsampling per tree (stochastic
                                   # gradient boosting) -- further
                                   # regularization via bagging-style
                                   # variance reduction
            colsample_bytree=0.8,  # feature subsampling per tree, same
                                   # rationale as subsample but across
                                   # the (highly correlated, since many
                                   # are lags of the same series)
                                   # feature columns
            reg_lambda=1.0,        # L2 regularization on leaf weights,
                                   # standard XGBoost ridge-style penalty
        )
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = fold_model.predict(X.iloc[val_idx])

        # MAE: mean absolute error, in the same units as water level (m)
        # -- directly interpretable ("the model is typically off by X
        # metres"). RMSE: root-mean-square error, penalizes large
        # individual errors more heavily than MAE (relevant for flood
        # forecasting, where a single badly-missed spike matters more
        # than many small errors).
        mae = mean_absolute_error(y.iloc[val_idx], pred)
        rmse = root_mean_squared_error(y.iloc[val_idx], pred)
        scores.append({"fold": fold, "mae": mae, "rmse": rmse})
        logger.info("xgb fold trained", extra=scores[-1])

    # Once cross-validation has produced an honest, walk-forward-style
    # performance estimate (the `scores` list above), refit a FINAL
    # model on the ENTIRE dataset for deployment. Using only the last
    # CV fold's model here would be a bug: TimeSeriesSplit's final fold
    # trains on all-but-the-last block, silently discarding the most
    # recent data from the deployed model even though it's the most
    # relevant to current conditions.
    final_model = xgb.XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    )
    final_model.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"xgb_{target_col}_{horizon_hours}h.joblib"
    joblib.dump({"model": final_model, "feature_cols": feature_cols}, model_path)

    return {"model_path": str(model_path), "cv_scores": scores}