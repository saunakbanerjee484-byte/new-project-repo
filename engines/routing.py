"""
engines/routing.py

Muskingum-Cunge flood-wave routing: a hydrologic (not full hydraulic)
routing method used for fast river-reach forecasting where a full
Saint-Venant solve (engines/dam_break.py) is unnecessary -- e.g. routing
an upstream hourly telemetry hydrograph down to a forecast point.
"""

import numpy as np


def muskingum_cunge_params(Q_ref, top_width_m, slope, celerity_mps, dx_m, dt_s):
    """
    Compute Muskingum K, X from physical reach properties (Cunge's
    hydraulic derivation, avoids needing calibrated K/X from historical
    hydrographs).

    K = dx / celerity
    X = 0.5 * (1 - Q_ref / (top_width_m * slope * celerity_mps * dx_m))
    """
    K = dx_m / celerity_mps
    X = 0.5 * (1 - Q_ref / (top_width_m * slope * celerity_mps * dx_m))
    X = float(np.clip(X, 0.0, 0.5))
    return K, X


def route_reach(inflow: np.ndarray, dt_s: float, K: float, X: float) -> np.ndarray:
    """
    Route an inflow hydrograph through a single reach via the standard
    Muskingum routing equation:
        Q_out[t+1] = C0*I[t+1] + C1*I[t] + C2*Q_out[t]
    """
    denom = 2 * K * (1 - X) + dt_s
    C0 = (dt_s - 2 * K * X) / denom
    C1 = (dt_s + 2 * K * X) / denom
    C2 = (2 * K * (1 - X) - dt_s) / denom
    assert abs(C0 + C1 + C2 - 1.0) < 1e-6, "Muskingum coefficients must sum to 1"

    outflow = np.zeros_like(inflow, dtype=float)
    outflow[0] = inflow[0]
    for t in range(len(inflow) - 1):
        outflow[t + 1] = C0 * inflow[t + 1] + C1 * inflow[t] + C2 * outflow[t]
    return outflow


def route_multi_reach(inflow: np.ndarray, dt_s: float, reach_params: list[tuple[float, float]]) -> np.ndarray:
    """Route through several sub-reaches in series, each with its own (K, X)."""
    flow = inflow
    for K, X in reach_params:
        flow = route_reach(flow, dt_s, K, X)
    return flow
