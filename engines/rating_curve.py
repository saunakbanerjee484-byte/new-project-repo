"""
engines/rating_curve.py

Stage-discharge rating curve calibration: Q = a * (h - h0) ** b,
fit via log-linear least squares over gauged (h, Q) pairs, with h0
(effective gauge-zero / bed level) refined by golden-section search
since the equation is only linear in log-space once h0 is fixed.
"""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize_scalar


@dataclass
class RatingCurve:
    a: float
    b: float
    h0: float

    def discharge(self, h):
        h = np.asarray(h, dtype=float)
        depth = np.maximum(h - self.h0, 0.0)
        return self.a * depth ** self.b

    def stage(self, Q, h_bounds=(0.0, 50.0)):
        """Invert numerically for stage given discharge (bisection)."""
        from scipy.optimize import brentq
        f = lambda h: self.discharge(h) - Q
        return brentq(f, self.h0 + 1e-3, h_bounds[1])


def _fit_ab_given_h0(h, Q, h0):
    depth = np.maximum(np.asarray(h) - h0, 1e-6)
    log_depth, log_Q = np.log(depth), np.log(np.asarray(Q))
    b, log_a = np.polyfit(log_depth, log_Q, 1)
    a = np.exp(log_a)
    resid = log_Q - (log_a + b * log_depth)
    sse = float(np.sum(resid ** 2))
    return a, b, sse


def calibrate_rating_curve(h, Q, h0_bounds=None) -> RatingCurve:
    """
    Calibrate Q = a(h - h0)^b from paired gauge readings and discharge
    measurements (e.g. current-meter gaugings).

    h0_bounds : search range for the effective gauge-zero (defaults to
                just below the lowest observed stage).
    """
    h, Q = np.asarray(h, dtype=float), np.asarray(Q, dtype=float)
    if h0_bounds is None:
        h0_bounds = (h.min() - 5.0, h.min() - 1e-3)

    def objective(h0):
        _, _, sse = _fit_ab_given_h0(h, Q, h0)
        return sse

    result = minimize_scalar(objective, bounds=h0_bounds, method="bounded")
    h0_opt = result.x
    a, b, _ = _fit_ab_given_h0(h, Q, h0_opt)
    return RatingCurve(a=float(a), b=float(b), h0=float(h0_opt))
