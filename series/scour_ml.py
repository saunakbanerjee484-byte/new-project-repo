"""
models/scour_ml.py

Prediction of Local Scour Around Bridge Piers: Empirical vs. Machine
Learning Models.
=======================================================================
TECHNICAL BACKGROUND (read before using):
FHWA's HEC-18/CSU equation (engines.bridge_torrents.hec18_pier_scour_depth)
is the industry-default formula for local pier scour, but it was fitted
mainly to controlled flume data and is well documented (Mueller & Jones,
1997, USGS) to over-predict scour in the field, especially for coarse
bed material. Recent literature (Etemad-Shahidi et al. 2015; Kim et al.
2024; Baranwal et al. 2024) shows tree-ensemble ML models (gradient
boosting, XGBoost) trained on the same governing dimensionless groups
outperform HEC-18 on held-out lab+field data while remaining physically
interpretable via SHAP analysis. A December 2025 study (Water journal,
NGBoost model) found the flow-intensity ratio V/Vc, relative depth y/b,
and Froude number Fr were the dominant SHAP-important features -- this
module's feature set below was chosen to match that literature rather
than an arbitrary set, and each formula is cited at the point of use.
"""

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score

# Reuse the already-verified HEC-18 empirical formula and Froude-number
# helper from engines.bridge_torrents, rather than re-implementing them
# here -- this module's job is the ML *comparison*, not re-deriving the
# baseline it benchmarks against.
from engines.bridge_torrents import hec18_pier_scour_depth, froude_number
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)
MODEL_PATH = settings.MODELS_DIR / "scour_gbr.joblib"

# SI-unit coefficient in Vc = Ku * y^(1/6) * D50^(1/3) (Ku = 11.17 in
# English/ft-s units; 6.19 is the SI/metric equivalent, confirmed
# against the FHWA HEC-18 manual and independent worked examples during
# the web search for this module).
KU_SI = 6.19

# The four dimensionless feature groups below are the ones repeatedly
# flagged as SHAP-important in the ML pier-scour literature reviewed for
# this module (Kim et al. 2024; the December 2025 NGBoost/CatBoost SHAP
# study): Froude number (flow regime), relative flow depth y1/a
# (geometry/force balance), flow-intensity ratio V1/Vc (how far above
# the sediment-motion threshold the approach flow sits), and relative
# grain coarseness d50/a (how "point-like" the sediment is versus the
# pier). Pier shape is retained as a categorical HEC-18-style correction
# since it is a first-order geometric control not captured by the
# hydraulic ratios above.
FEATURE_COLUMNS = ["froude_number", "y1_over_a", "flow_intensity_v_vc", "d50_over_a", "pier_shape_factor"]


def hec18_critical_velocity(y1_m, d50_m, Ku=KU_SI):
    """
    Laursen/Neill competent (critical) velocity: the depth-averaged flow
    velocity at which the D50 bed-material grain size just begins to
    move -- HEC-18's own criterion for whether a site is live-bed
    (V1 >= Vc, bed material already in general motion) or clear-water
    (V1 < Vc, only the local pier-induced vortex disturbs the bed)
    scour, and the denominator of the V1/Vc flow-intensity feature used
    below.

    THEORY / FORMULA (HEC-18 Equation, after Laursen 1963 / Neill 1968;
    confirmed via web search against the FHWA HEC-18 manual and multiple
    independent worked examples during the writing of this module):

        Vc = Ku * y1^(1/6) * D50^(1/3)             (SI: Ku = 6.19, m & s)

    Both y1 (approach depth) and D50 (median grain size) are in metres;
    the 1/6 and 1/3 exponents come from combining a Strickler-type
    roughness relation for the bed with the Shields-based threshold
    shear stress -- HEC-18 presents this as a single empirical
    coefficient (Ku) rather than deriving it from Shields directly, so
    it is reproduced here exactly as published rather than re-derived
    from engines.sediment's Shields-based approach (the two are related
    but not numerically identical without additional simplifying
    assumptions HEC-18 itself makes).
    """
    # Direct application of the Laursen/Neill formula above -- y1 and
    # d50 both enter as fractional powers (1/6 and 1/3 respectively),
    # so the result already carries m/s units when y1, d50 are in metres.
    return Ku * y1_m ** (1.0 / 6.0) * d50_m ** (1.0 / 3.0)


@dataclass
class ScourDataset:
    """
    Container for a labeled pier-scour dataset (flume or field
    measurements), used both to train the ML model and to benchmark it
    against the HEC-18 empirical baseline on the same rows.

    Expected columns in `df`:
        y1                  : approach flow depth (m)
        V1                  : approach mean velocity (m/s)
        pier_width          : pier width `a` (m)
        d50                 : bed-material median grain size (m)
        pier_shape_factor   : HEC-18 K1 nose-shape correction
                               (1.0 round/circular nose, 1.1 square nose)
        observed_scour_depth: measured scour depth (m) -- the ML target
    """
    df: pd.DataFrame

    def to_features(self) -> pd.DataFrame:
        """
        Converts the raw physical measurements above into the
        dimensionless feature groups in FEATURE_COLUMNS -- ML models
        trained on dimensionless groups generalize far better across
        scale (lab-flume cm-scale piers vs field metre-scale piers) than
        ones trained on raw dimensional values, since the underlying
        physics (Shields mobility, Froude similarity) is itself
        dimensionless.
        """
        out = self.df.copy()

        # Fr1 = V1 / sqrt(g*y1): approach-flow Froude number, governs
        # whether the bow wave/horseshoe vortex system at the pier is
        # sub- or super-critical -- reused from engines.bridge_torrents
        # rather than recomputed here.
        out["froude_number"] = froude_number(out["V1"], out["y1"])

        # y1/a: relative flow depth -- ratio of approach depth to pier
        # width. Controls whether the horseshoe vortex (which scales
        # with pier width) is "depth-limited" (shallow flow, y1/a small)
        # or fully developed (deep flow, y1/a large) -- the primary
        # SHAP-important variable in the December 2025 NGBoost study
        # reviewed for this module.
        out["y1_over_a"] = out["y1"] / out["pier_width"]

        # V1/Vc: flow-intensity ratio -- how far the approach velocity
        # sits above the Laursen/Neill sediment-motion threshold
        # computed in hec18_critical_velocity() above. V1/Vc < 1 is
        # clear-water scour (HEC-18's own live-bed/clear-water
        # criterion); V1/Vc > 1 is live-bed. Flagged in the literature
        # (Kim et al. 2024) as a dominant SHAP feature alongside y1/a.
        vc = hec18_critical_velocity(out["y1"], out["d50"])
        out["flow_intensity_v_vc"] = out["V1"] / vc

        # d50/a: relative grain coarseness -- ratio of median bed-grain
        # size to pier width. A small d50/a means the sediment is
        # effectively a continuum relative to the pier (fine sand around
        # a wide pier); a larger d50/a means individual grains are a
        # non-negligible fraction of the pier's scale (coarse gravel
        # around a narrow pier), which changes scour-hole geometry.
        out["d50_over_a"] = out["d50"] / out["pier_width"]

        return out


def empirical_predictions(dataset: ScourDataset) -> np.ndarray:
    """
    Runs the HEC-18/CSU formula (engines.bridge_torrents.
    hec18_pier_scour_depth) row-by-row over the dataset, using each
    row's own pier_shape_factor as the HEC-18 K1 correction -- this is
    the baseline that train_scour_model() below benchmarks the ML model
    against, evaluated on the *same* rows for a fair, paired comparison.
    """
    df = dataset.df
    # Row-wise HEC-18 evaluation: ys/y1 = 2*K1*K2*K3*K4*(a/y1)^0.65*Fr1^0.43
    # (see engines/bridge_torrents.py for the full formula and citation);
    # here K2, K3, K4 are left at their HEC-18 default values (1.0, 1.1,
    # 1.0) since this module's dataset schema only carries the pier-nose
    # shape correction (K1) explicitly.
    return np.array([
        hec18_pier_scour_depth(row.y1, row.V1, row.pier_width, K1=row.pier_shape_factor)
        for row in df.itertuples()
    ])


def train_scour_model(dataset: ScourDataset, n_splits: int = 5) -> dict:
    """
    Trains a gradient-boosted regression tree ensemble on the
    dimensionless features above and benchmarks it against the HEC-18
    empirical baseline, using k-fold cross-validated predictions for
    the ML side so both models are compared on genuinely held-out data
    (HEC-18 needs no "training", so it is simply evaluated on every row;
    the ML comparison would be unfair if evaluated on its own training
    rows, hence cross_val_predict rather than in-sample .predict()).

    THEORY: gradient boosting (Friedman, 2001) builds an ensemble of
    shallow decision trees sequentially, each new tree fit to the
    residual errors of the ensemble so far -- well suited to the
    piecewise-nonlinear, interaction-heavy relationship between
    Fr/y1-a/V1-Vc/d50-a and scour depth that the literature reviewed for
    this module (SHAP analyses) shows is not a simple power law of the
    kind HEC-18 assumes.
    """
    feat_df = dataset.to_features()
    X, y = feat_df[FEATURE_COLUMNS], feat_df["observed_scour_depth"]

    # Shallow trees (max_depth=3) + many of them (300) + a small
    # learning rate (0.05) is the standard gradient-boosting recipe for
    # avoiding overfitting on modest-sized scour datasets (typically a
    # few hundred to a few thousand rows in the published lab+field
    # compilations) while still capturing feature interactions.
    model = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05)

    # K-fold cross-validation: split the data into n_splits folds, train
    # on n_splits-1 of them and predict the held-out fold, rotating
    # through all folds -- cross_val_predict returns one out-of-fold
    # prediction per row, giving an honest (non-overfit) MAE/R2 estimate
    # for the ML model, directly comparable to HEC-18's in-sample
    # evaluation (HEC-18 has no fold structure since it isn't fitted to
    # this data at all).
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_pred = cross_val_predict(model, X, y, cv=cv)

    # Refit on the FULL dataset for the final deployed model (the
    # cross-validation above was only for scoring; once we trust the
    # score, using all available data for the saved model is standard
    # practice, since cross_val_predict's per-fold models are discarded).
    model.fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    empirical_pred = empirical_predictions(dataset)

    # Paired comparison: mean absolute error (MAE, same units as scour
    # depth -- directly interpretable) and R^2 (fraction of variance
    # explained) for both the ML model (out-of-fold) and HEC-18
    # (in-sample, since it has no folds) against the same observed
    # scour-depth column.
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
    """
    Inference-time wrapper: loads the trained model from MODEL_PATH and
    predicts scour depth for a single new pier/flow condition, computing
    the same dimensionless features used at training time (Fr1, y1/a,
    V1/Vc, d50/a) so the model sees inputs in the same space it learned.
    """
    model = joblib.load(MODEL_PATH)

    # Same four feature computations as ScourDataset.to_features() above,
    # applied to this single new observation rather than a DataFrame
    # column, since there is no dataset object at inference time.
    fr = froude_number(V1, y1)
    vc = hec18_critical_velocity(y1, d50)
    X = pd.DataFrame([{
        "froude_number": fr,
        "y1_over_a": y1 / pier_width,
        "flow_intensity_v_vc": V1 / vc,
        "d50_over_a": d50 / pier_width,
        "pier_shape_factor": pier_shape_factor,
    }])
    return float(model.predict(X)[0])