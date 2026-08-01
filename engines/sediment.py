"""
engines/sediment.py

Sediment mobility via the Shields parameter (dimensionless critical
shear stress) and bed-material incipient-motion checks -- feeds both
general channel-stability assessment and the scour engines
(bridge_torrents.py, models/scour_ml.py).
"""

import numpy as np

G = 9.81
RHO_WATER = 1000.0
RHO_SEDIMENT = 2650.0  # quartz sand/gravel, typical alluvial value


def shields_parameter(tau_bed_pa, d50_m, rho_s=RHO_SEDIMENT, rho_w=RHO_WATER):
    """theta = tau / ((rho_s - rho_w) * g * d50)"""
    return tau_bed_pa / ((rho_s - rho_w) * G * d50_m)


def bed_shear_stress(depth_m, slope, rho_w=RHO_WATER):
    """tau = rho * g * h * S (normal-flow approximation)."""
    return rho_w * G * depth_m * slope


def critical_shields_number(d_star: float) -> float:
    """
    Soulsby (1997) explicit fit for critical Shields number as a function
    of dimensionless grain size D*, avoiding iterative Shields-diagram
    lookup.
    """
    return 0.30 / (1 + 1.2 * d_star) + 0.055 * (1 - np.exp(-0.020 * d_star))


def dimensionless_grain_size(d50_m, rho_s=RHO_SEDIMENT, rho_w=RHO_WATER, nu=1.0e-6):
    """D* = d50 * [(rho_s/rho_w - 1) * g / nu^2] ^ (1/3)"""
    return d50_m * ((rho_s / rho_w - 1) * G / nu ** 2) ** (1 / 3)


def is_bed_mobile(depth_m, slope, d50_m) -> dict:
    """Compare actual vs. critical Shields parameter for incipient motion."""
    tau = bed_shear_stress(depth_m, slope)
    theta = shields_parameter(tau, d50_m)
    d_star = dimensionless_grain_size(d50_m)
    theta_cr = critical_shields_number(d_star)
    return {
        "tau_bed_pa": tau,
        "shields_parameter": theta,
        "critical_shields_number": theta_cr,
        "mobile": bool(theta > theta_cr),
        "mobility_ratio": theta / theta_cr,
    }
