"""
DeltaPulse / FlowState - Real-Time Flood Intelligence & River Analytics Platform
Module: wb_flood_intelligence/models/forecaster.py
Description: Production-Grade Multi-Horizon Water Level Inference Engine (6h and 12h Lead Times).

CONSTRAINTS & SPECIFICATIONS:
- Strictly NO TensorFlow. Uses PyTorch, XGBoost, or Physics-Informed Kinematic ARX Fallback.
- Serves real-time water level forecasts for West Bengal stations (e.g., Durgapur Barrage,
  Panchet Reservoir, Teesta Bridge Jalpaiguri, Kolaghat Rupnarayan).
- Evaluates predicted levels against district Warning Levels (WL) & Danger Levels (DL).
- Emits downstream risk triggers for SWMM Urban Surcharge, Dam-Break Wave Arrivals,
  Bridge Pier Scour Depths, and Embankment Seepage Heads.
"""

import os
import sys
import json
import logging
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Configure Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("DeltaPulse.Forecaster")

# Framework Detection (Excluding TensorFlow)
HAS_TORCH = False
HAS_XGBOOST = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    logger.info("PyTorch runtime detected.")
except ImportError:
    logger.info("PyTorch runtime not detected. PyTorch LSTM inference disabled.")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
    logger.info("XGBoost runtime detected.")
except ImportError:
    logger.info("XGBoost runtime not detected. XGBoost inference disabled.")


# =====================================================================
# DATA CLASSES & TYPINGS FOR INFERENCE OUTPUT
# =====================================================================

@dataclass
class HorizonPrediction:
    """Prediction details for a single forecast horizon (+6h or +12h)."""
    horizon_hours: int
    target_timestamp: str
    predicted_water_level_m: float
    confidence_lower_95_m: float
    confidence_upper_95_m: float
    projected_rate_of_rise_m_hr: float
    status_flag: str                     # NORMAL, ELEVATED, WARNING, DANGER
    warning_breached: bool
    danger_breached: bool
    estimated_discharge_cumes: float     # Derived via Rating Curve Q = a(h-h0)^b


@dataclass
class SecondaryRiskIndicators:
    """Downstream coupled module triggers & hydraulic risk indicators."""
    swmm_outfall_surcharge_risk: bool
    swmm_backwater_head_m: float
    estimated_bridge_pier_scour_depth_m: float
    embankment_seepage_gradient_alert: bool
    dam_break_surge_wave_risk: bool


@dataclass
class ForecastResult:
    """Complete Inference Output Object returned by WaterLevelForecaster."""
    station_id: str
    station_name: str
    river: str
    district: str
    inference_timestamp: str
    current_water_level_m: float
    warning_level_m: float
    danger_level_m: float
    model_type_used: str                 # 'PyTorch-LSTM', 'XGBoost', or 'Physics-Informed-ARX'
    forecast_6h: HorizonPrediction
    forecast_12h: HorizonPrediction
    secondary_risks: SecondaryRiskIndicators
    execution_time_ms: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# =====================================================================
# PYTORCH LSTM ARCHITECTURE (NO TENSORFLOW)
# =====================================================================

if HAS_TORCH:
    class PyTorchWaterLevelLSTM(nn.Module):
        """
        Deep Recurrent Neural Network for Multi-Horizon Water Level Forecasting.
        Inputs: Window of historical features [Batch, Seq_Len, Num_Features].
        Outputs: 2 Continuous values [Level_t+6h, Level_t+12h].
        """
        def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
            super(PyTorchWaterLevelLSTM, self).__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            self.fc1 = nn.Linear(hidden_dim, 32)
            self.relu = nn.ReLU()
            self.out_head = nn.Linear(32, 2)  # Output [Level_+6h, Level_+12h]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            lstm_out, _ = self.lstm(x)
            last_hidden = lstm_out[:, -1, :]
            dense = self.relu(self.fc1(last_hidden))
            return self.out_head(dense)


# =====================================================================
# PHYSICS-INFORMED ARX ENGINE (STANDALONE FALLBACK)
# =====================================================================

class PhysicsInformedARXEngine:
    """
    Hydraulic-Statistical Forecaster combining Autoregressive Exogenous (ARX)
    dynamics with kinematic wave propagation and stage-discharge rating curves.
    """

    def __init__(self, rating_a: float = 45.2, rating_h0: float = 200.0, rating_b: float = 1.65):
        self.a = rating_a
        self.h0 = rating_h0
        self.b = rating_b

    def stage_to_discharge(self, h: float) -> float:
        """Computes discharge Q from water level h via Q = a * (h - h0)^b."""
        head = max(0.001, h - self.h0)
        return float(self.a * (head ** self.b))

    def discharge_to_stage(self, Q: float) -> float:
        """Inverse rating curve: h = h0 + (Q / a)^(1/b)."""
        Q_safe = max(0.001, Q)
        return float(self.h0 + (Q_safe / self.a) ** (1.0 / self.b))

    def predict_horizons(
        self,
        current_h: float,
        dh_dt_1h: float,
        dh_dt_3h: float,
        recent_rain_6h_mm: float,
        upstream_discharge_cumes: float
    ) -> Tuple[float, float, float, float]:
        """
        Calculates +6h and +12h forecasted water levels and rate of rise.
        Returns: (pred_6h, pred_12h, rate_6h, rate_12h)
        """
        decay_6h = math.exp(-0.08 * 6)
        decay_12h = math.exp(-0.08 * 12)

        runoff_coef = 0.012  # Runoff-to-stage conversion (mm rain -> m stage rise)

        current_Q = self.stage_to_discharge(current_h)
        Q_diff = upstream_discharge_cumes - current_Q
        upstream_impact_6h = (Q_diff * 0.00015) * (1 - decay_6h)
        upstream_impact_12h = (Q_diff * 0.00028) * (1 - decay_12h)

        delta_h_6h = (dh_dt_3h * 6.0 * decay_6h) + (recent_rain_6h_mm * runoff_coef) + upstream_impact_6h
        delta_h_12h = (dh_dt_3h * 12.0 * decay_12h) + (recent_rain_6h_mm * runoff_coef * 1.5) + upstream_impact_12h

        pred_6h = max(self.h0 + 0.1, current_h + delta_h_6h)
        pred_12h = max(self.h0 + 0.1, current_h + delta_h_12h)

        rate_6h = (pred_6h - current_h) / 6.0
        rate_12h = (pred_12h - current_h) / 12.0

        return float(pred_6h), float(pred_12h), float(rate_6h), float(rate_12h)


# =====================================================================
# MAIN INFERENCE FORECASTER CLASS
# =====================================================================

class WaterLevelForecaster:
    """
    Main Inference Engine for West Bengal Flood Intelligence (DeltaPulse).
    Generates 6-hour and 12-hour water level forecasts and risk alerts.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        model_dir: Optional[str] = None,
        preferred_engine: str = "AUTO"  # 'AUTO', 'PYTORCH', 'XGBOOST', 'PHYSICS'
    ):
        self.project_root = Path(__file__).resolve().parent.parent
        self.config_path = Path(config_path) if config_path else self.project_root / "config" / "thresholds.json"
        self.model_dir = Path(model_dir) if model_dir else self.project_root / "models"
        self.preferred_engine = preferred_engine.upper()

        self.thresholds = self._load_thresholds()
        self.active_engine_name = "Physics-Informed-ARX"
        self.pytorch_model = None
        self.xgb_model_6h = None
        self.xgb_model_12h = None
        self.physics_engines: Dict[str, PhysicsInformedARXEngine] = {}

        self._initialize_engines()

    def _load_thresholds(self) -> Dict[str, dict]:
        """Loads station warning & danger thresholds from JSON configuration."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found at {self.config_path}. Using internal defaults.")
            return {
                "STATION_DURGAPUR_BARRAGE": {
                    "station_name": "Durgapur Barrage Downstream",
                    "district": "Paschim Bardhaman",
                    "river": "Damodar",
                    "warning_level_m": 208.50,
                    "danger_level_m": 211.00,
                    "zero_gauge_m": 200.00,
                    "rating_curve_params": {"a": 45.2, "h0": 200.0, "b": 1.65}
                }
            }
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading thresholds: {e}")
            return {}

    def _initialize_engines(self):
        """Attempts to load trained PyTorch or XGBoost models; defaults to Physics Engine."""
        if (self.preferred_engine in ["AUTO", "PYTORCH"]) and HAS_TORCH:
            pt_weights = self.model_dir / "lstm_water_level_wb.pt"
            if pt_weights.exists():
                try:
                    self.pytorch_model = PyTorchWaterLevelLSTM(input_dim=15, hidden_dim=64)
                    self.pytorch_model.load_state_dict(torch.load(pt_weights, map_location="cpu"))
                    self.pytorch_model.eval()
                    self.active_engine_name = "PyTorch-LSTM"
                    logger.info("Loaded PyTorch LSTM model weights.")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load PyTorch weights: {e}")

        if (self.preferred_engine in ["AUTO", "XGBOOST"]) and HAS_XGBOOST:
            xgb_path_6h = self.model_dir / "xgb_model_6h.json"
            xgb_path_12h = self.model_dir / "xgb_model_12h.json"
            if xgb_path_6h.exists() and xgb_path_12h.exists():
                try:
                    self.xgb_model_6h = xgb.Booster()
                    self.xgb_model_6h.load_model(str(xgb_path_6h))
                    self.xgb_model_12h = xgb.Booster()
                    self.xgb_model_12h.load_model(str(xgb_path_12h))
                    self.active_engine_name = "XGBoost"
                    logger.info("Loaded XGBoost model artifacts.")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load XGBoost artifacts: {e}")

        self.active_engine_name = "Physics-Informed-ARX"
        logger.info("Initialized Physics-Informed Kinematic Wave & ARX Fallback Engine.")

    def _get_physics_engine_for_station(self, station_id: str) -> PhysicsInformedARXEngine:
        if station_id not in self.physics_engines:
            st_cfg = self.thresholds.get(station_id, {})
            rc = st_cfg.get("rating_curve_params", {"a": 45.2, "h0": 200.0, "b": 1.65})
            self.physics_engines[station_id] = PhysicsInformedARXEngine(
                rating_a=rc["a"], rating_h0=rc["h0"], rating_b=rc["b"]
            )
        return self.physics_engines[station_id]

    def _determine_status(self, predicted_level: float, warning_lvl: float, danger_lvl: float) -> str:
        if predicted_level >= danger_lvl:
            return "DANGER"
        elif predicted_level >= warning_lvl:
            return "WARNING"
        elif predicted_level >= warning_lvl - 0.5:
            return "ELEVATED"
        return "NORMAL"

    def predict(
        self,
        telemetry_df: pd.DataFrame,
        station_id: str = "STATION_DURGAPUR_BARRAGE"
    ) -> ForecastResult:
        """Executes real-time inference for 6-hour and 12-hour water level predictions."""
        start_time = datetime.now()

        # 1. Retrieve Station Metadata
        station_info = self.thresholds.get(station_id, {
            "station_name": f"Station {station_id}",
            "district": "West Bengal Region",
            "river": "Unknown River",
            "warning_level_m": 208.50,
            "danger_level_m": 211.00,
            "zero_gauge_m": 200.00,
            "rating_curve_params": {"a": 45.2, "h0": 200.0, "b": 1.65}
        })

        warning_lvl = station_info.get("warning_level_m", 208.50)
        danger_lvl = station_info.get("danger_level_m", 211.00)

        if telemetry_df.empty or len(telemetry_df) < 6:
            raise ValueError("telemetry_df must contain at least 6 hourly records.")

        # 2. Extract Current State
        latest_row = telemetry_df.iloc[-1]
        current_h = float(latest_row["water_level_m"])
        current_time = latest_row.name if isinstance(latest_row.name, (pd.Timestamp, datetime)) else datetime.now()

        h_series = telemetry_df["water_level_m"].values
        dh_dt_1h = float(h_series[-1] - h_series[-2]) if len(h_series) >= 2 else 0.0
        dh_dt_3h = float((h_series[-1] - h_series[-4]) / 3.0) if len(h_series) >= 4 else dh_dt_1h

        recent_rain_6h = float(telemetry_df["rainfall_mm"].iloc[-6:].sum()) if "rainfall_mm" in telemetry_df.columns else 0.0
        upstream_Q = float(telemetry_df["upstream_discharge_cumes"].iloc[-1]) if "upstream_discharge_cumes" in telemetry_df.columns else 1200.0

        phys_engine = self._get_physics_engine_for_station(station_id)

        # 3. Model Inference Execution
        pred_6h, pred_12h = current_h, current_h
        rate_6h, rate_12h = 0.0, 0.0
        used_model = self.active_engine_name

        if self.active_engine_name == "PyTorch-LSTM" and self.pytorch_model is not None and HAS_TORCH:
            try:
                seq_data = telemetry_df.iloc[-24:].values
                tensor_in = torch.tensor(seq_data, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    out = self.pytorch_model(tensor_in).numpy()[0]
                pred_6h, pred_12h = float(out[0]), float(out[1])
                rate_6h = (pred_6h - current_h) / 6.0
                rate_12h = (pred_12h - current_h) / 12.0
            except Exception as e:
                logger.warning(f"PyTorch prediction failed ({e}). Falling back to Physics ARX.")
                used_model = "Physics-Informed-ARX"

        if used_model == "Physics-Informed-ARX":
            pred_6h, pred_12h, rate_6h, rate_12h = phys_engine.predict_horizons(
                current_h=current_h,
                dh_dt_1h=dh_dt_1h,
                dh_dt_3h=dh_dt_3h,
                recent_rain_6h_mm=recent_rain_6h,
                upstream_discharge_cumes=upstream_Q
            )

        # 4. Error Bounds & Rating Curve Discharge Conversions
        std_6h, std_12h = 0.18, 0.35
        Q_6h = phys_engine.stage_to_discharge(pred_6h)
        Q_12h = phys_engine.stage_to_discharge(pred_12h)

        time_6h_str = (current_time + timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%S")
        time_12h_str = (current_time + timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%S")

        # 5. Build Horizon Prediction Objects
        forecast_6h_obj = HorizonPrediction(
            horizon_hours=6,
            target_timestamp=time_6h_str,
            predicted_water_level_m=round(pred_6h, 3),
            confidence_lower_95_m=round(pred_6h - 1.96 * std_6h, 3),
            confidence_upper_95_m=round(pred_6h + 1.96 * std_6h, 3),
            projected_rate_of_rise_m_hr=round(rate_6h, 4),
            status_flag=self._determine_status(pred_6h, warning_lvl, danger_lvl),
            warning_breached=(pred_6h >= warning_lvl),
            danger_breached=(pred_6h >= danger_lvl),
            estimated_discharge_cumes=round(Q_6h, 2)
        )

        forecast_12h_obj = HorizonPrediction(
            horizon_hours=12,
            target_timestamp=time_12h_str,
            predicted_water_level_m=round(pred_12h, 3),
            confidence_lower_95_m=round(pred_12h - 1.96 * std_12h, 3),
            confidence_upper_95_m=round(pred_12h + 1.96 * std_12h, 3),
            projected_rate_of_rise_m_hr=round(rate_12h, 4),
            status_flag=self._determine_status(pred_12h, warning_lvl, danger_lvl),
            warning_breached=(pred_12h >= warning_lvl),
            danger_breached=(pred_12h >= danger_lvl),
            estimated_discharge_cumes=round(Q_12h, 2)
        )

        # 6. Coupled Secondary Risk Assessments
        max_future_level = max(pred_6h, pred_12h)
        swmm_surcharge = (max_future_level > (warning_lvl + 0.3))
        backwater_head = max(0.0, max_future_level - warning_lvl)

        flow_depth = max(0.5, pred_12h - station_info.get("zero_gauge_m", 200.0))
        est_velocity = min(5.5, Q_12h / (flow_depth * 45.0))
        froude_num = est_velocity / math.sqrt(9.81 * flow_depth)
        scour_depth = 2.0 * 2.5 * 1.1 * 1.0 * (froude_num ** 0.65) if froude_num > 0 else 0.5

        embankment_risk = (max_future_level >= danger_lvl)
        dam_break_risk = (dh_dt_3h > 0.4 or upstream_Q > 8000.0)

        secondary_risks = SecondaryRiskIndicators(
            swmm_outfall_surcharge_risk=swmm_surcharge,
            swmm_backwater_head_m=round(backwater_head, 3),
            estimated_bridge_pier_scour_depth_m=round(scour_depth, 2),
            embankment_seepage_gradient_alert=embankment_risk,
            dam_break_surge_wave_risk=dam_break_risk
        )

        exec_time = (datetime.now() - start_time).total_seconds() * 1000.0

        # 7. Final Return Statement
        return ForecastResult(
            station_id=station_id,
            station_name=station_info.get("station_name", "Unknown Station"),
            river=station_info.get("river", "Unknown River"),
            district=station_info.get("district", "West Bengal"),
            inference_timestamp=current_time.strftime("%Y-%m-%dT%H:%M:%S") if isinstance(current_time, (datetime, pd.Timestamp)) else str(current_time),
            current_water_level_m=round(current_h, 3),
            warning_level_m=warning_lvl,
            danger_level_m=danger_lvl,
            model_type_used=used_model,
            forecast_6h=forecast_6h_obj,
            forecast_12h=forecast_12h_obj,
            secondary_risks=secondary_risks,
            execution_time_ms=round(exec_time, 2)
        )