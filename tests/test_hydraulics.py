"""
tests/test_hydraulics.py

Verify rating-curve calibration and Muskingum-Cunge routing math.
"""

import numpy as np

from engines.rating_curve import calibrate_rating_curve
from engines.routing import muskingum_cunge_params, route_reach
from engines.sediment import shields_parameter, bed_shear_stress


def test_rating_curve_recovers_known_params():
    true_a, true_b, true_h0 = 12.5, 1.6, 1.2
    h = np.linspace(2.0, 8.0, 15)
    Q = true_a * (h - true_h0) ** true_b

    rc = calibrate_rating_curve(h, Q)
    Q_pred = rc.discharge(h)
    assert np.allclose(Q_pred, Q, rtol=0.05)


def test_muskingum_routing_conserves_volume_for_steady_inflow():
    K, X = muskingum_cunge_params(
        Q_ref=500, top_width_m=80, slope=0.0006, celerity_mps=1.8, dx_m=5000, dt_s=3600
    )
    inflow = np.full(50, 500.0)
    outflow = route_reach(inflow, dt_s=3600, K=K, X=X)
    assert np.isclose(outflow[-1], inflow[-1], rtol=0.02)


def test_shields_parameter_increases_with_shear():
    tau_low = bed_shear_stress(depth_m=1.0, slope=0.0005)
    tau_high = bed_shear_stress(depth_m=3.0, slope=0.0015)
    theta_low = shields_parameter(tau_low, d50_m=0.002)
    theta_high = shields_parameter(tau_high, d50_m=0.002)
    assert theta_high > theta_low
