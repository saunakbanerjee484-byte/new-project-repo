"""
engines/bridge_torrents.py

Supercritical (torrential) open-channel flow analysis and bridge-pier
protection sizing for the upper/hill-course reaches of West Bengal rivers
(e.g. Teesta, upper Damodar tributaries) where steep slopes routinely
push flow past Fr = 1, unlike the subcritical lower/deltaic reaches.

Covers:
- Froude-number flow-regime classification & critical depth/energy
- Hydraulic jump conjugate depth (where torrential flow decelerates
  into a barrage pool / bridge waterway -- governs stilling-basin and
  pier-approach design)
- HEC-18 (CSU) local pier-scour depth -- the standard empirical formula,
  used as the baseline that models/scour_ml.py benchmarks an ML model
  against
- Riprap (rock armor) sizing for pier protection against that scour
"""

import numpy as np

G = 9.81


def froude_number(velocity_mps, depth_m):
    return velocity_mps / np.sqrt(G * np.maximum(depth_m, 1e-9))


def flow_regime(velocity_mps, depth_m) -> str:
    fr = froude_number(velocity_mps, depth_m)
    if fr < 0.95:
        return "subcritical"
    if fr > 1.05:
        return "supercritical (torrential)"
    return "near-critical (transitional -- avoid siting piers here)"


def critical_depth_rectangular(q_unit_width, g=G):
    """Critical depth for a unit-width discharge q (m^2/s): hc = (q^2/g)^(1/3)."""
    return (q_unit_width ** 2 / g) ** (1 / 3)


def specific_energy(depth_m, velocity_mps, g=G):
    return depth_m + velocity_mps ** 2 / (2 * g)


def hydraulic_jump_conjugate_depth(h1, fr1):
    """
    Conjugate (sequent) depth after a hydraulic jump, from the
    Belanger equation -- used to size stilling basins where a
    torrential approach flow must decelerate before a bridge/barrage.
    """
    return 0.5 * h1 * (np.sqrt(1 + 8 * fr1 ** 2) - 1)


def hec18_pier_scour_depth(y1, V1, pier_width, K1=1.0, K2=1.0, K3=1.1, K4=1.0):
    """
    HEC-18 (CSU) equation for local live-bed scour depth at a bridge
    pier -- the standard empirical baseline used by transportation
    hydraulics practice (FHWA HEC-18):

        ys / y1 = 2.0 * K1*K2*K3*K4 * (a/y1)^0.65 * Fr1^0.43

    Parameters
    ----------
    y1 : approach flow depth (m)
    V1 : approach velocity (m/s)
    pier_width : pier width `a` (m)
    K1 : pier nose shape (1.0 round nose/circular, 1.1 square nose)
    K2 : angle of attack correction (1.0 if flow-aligned)
    K3 : bed condition correction (1.1 typical live-bed plane bed)
    K4 : armoring correction for coarse bed material (1.0 if not armored)

    Returns
    -------
    ys : scour depth (m), measured below the ambient bed
    """
    Fr1 = froude_number(V1, y1)
    ys = y1 * 2.0 * K1 * K2 * K3 * K4 * (pier_width / y1) ** 0.65 * Fr1 ** 0.43
    return float(ys)


def riprap_d50_for_pier(V1, specific_gravity=2.65, safety_factor=1.0):
    """
    Riprap median stone size for pier scour protection, per the
    Isbash-type relation commonly used in FHWA HEC-23:
        d50 = 0.692 * V1^2 / [ (Ss - 1) * 2g ]  (SI, at-pier sizing)
    """
    d50 = 0.692 * V1 ** 2 / ((specific_gravity - 1) * 2 * G)
    return float(d50 * safety_factor)


def pier_protection_design(y1, V1, pier_width, **hec18_kwargs) -> dict:
    """Combined scour + riprap protection summary for a single pier."""
    regime = flow_regime(V1, y1)
    ys = hec18_pier_scour_depth(y1, V1, pier_width, **hec18_kwargs)
    d50 = riprap_d50_for_pier(V1)
    return {
        "flow_regime": regime,
        "froude_number": float(froude_number(V1, y1)),
        "predicted_scour_depth_m": ys,
        "recommended_riprap_d50_m": d50,
        "recommended_footing_depth_m": ys * 1.3,  # common 30% freeboard-on-scour practice
    }
