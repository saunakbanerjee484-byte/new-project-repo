"""
engines/embankment.py

Durability and seepage analysis of earth dams / embankments: phreatic
(seepage) line location via Casagrande's parabolic method, exit
gradient, and piping-safety checks -- the geotechnical counterpart to
the breach hydraulics in engines/dam_break.py (a seepage-driven "piping"
failure is one of the two Froehlich failure_mode options used there).
"""

import numpy as np

GAMMA_WATER = 9.81  # kN/m^3


def casagrande_phreatic_line(x, dam_height, upstream_water_depth, base_width,
                              downstream_slope_h_per_v):
    """
    Casagrande's parabolic approximation of the phreatic (seepage) line
    through a homogeneous earth embankment with no internal drain,
    origin at the upstream toe.

    Returns z(x): seepage-line elevation above the base at horizontal
    position x (m), using the basic parabola y^2 = 2*y0*x + y0^2 with
    Casagrande's entrance correction near the upstream face.
    """
    x = np.asarray(x, dtype=float)
    h = upstream_water_depth
    # Distance from the focus (downstream toe) to directrix, per Casagrande
    d = base_width - downstream_slope_h_per_v * 0.3 * dam_height  # approx horizontal projection
    y0 = np.sqrt(d ** 2 + h ** 2) - d  # parabola focal parameter
    z = np.sqrt(y0 ** 2 + 2 * y0 * np.maximum(x, 0)) 
    return np.minimum(z, h)


def exit_gradient(head_loss_m, exit_path_length_m):
    """Hydraulic (exit) gradient at the downstream toe/exit face."""
    return head_loss_m / max(exit_path_length_m, 1e-6)


def critical_gradient(specific_gravity_solids=2.65, void_ratio=0.6):
    """Terzaghi critical hydraulic gradient for piping onset: ic = (Gs-1)/(1+e)."""
    return (specific_gravity_solids - 1) / (1 + void_ratio)


def piping_safety_factor(exit_grad, ic=None, **ic_kwargs) -> dict:
    """
    Factor of safety against piping/heave = ic / i_exit. FoS < 1.5-2.0 is
    conventionally flagged for remedial filter/drain design (IS 8237 /
    USBR practice); FoS < 1.0 indicates active piping risk.
    """
    ic = critical_gradient(**ic_kwargs) if ic is None else ic
    fos = ic / max(exit_grad, 1e-9)
    return {
        "exit_gradient": exit_grad,
        "critical_gradient": ic,
        "factor_of_safety": fos,
        "status": ("safe" if fos >= 2.0 else "marginal" if fos >= 1.0 else "piping_risk"),
    }


def seepage_discharge_per_unit_length(k_mps, y0, dam_height):
    """
    Approximate seepage discharge per unit length of embankment through
    the phreatic surface: q = k * y0 (Casagrande/Schaffernak-derived
    approximation, y0 = focal parameter from casagrande_phreatic_line).
    """
    return k_mps * y0
