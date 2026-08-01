"""
engines/bridge_torrents.py

Supercritical (Torrential) Flow & Upper-Course Bridge-Pier Protection.

Analysis for the upper/hill-course reaches of West Bengal rivers (e.g.
Teesta, upper Damodar tributaries) where steep slopes routinely push
flow past Fr = 1, unlike the subcritical lower/deltaic reaches -- plus
the bridge-pier scour and protection design that depends on it.

Covers:
1. Froude-number flow-regime classification, critical depth/specific
   energy, and Manning normal-depth -> channel slope-type
   classification (steep/S vs mild/M vs critical/C), which determines
   whether a reach's *normal* flow is supercritical at all.
2. Ordinary (normal) hydraulic jump conjugate depth -- where torrential
   flow decelerates into a barrage pool or bridge waterway.
3. Bridge-opening choking check for a supercritical approach flow
   (minimum specific energy through a contraction) -- the classic
   torrential-river bridge-siting hazard: an undersized opening forces
   a hydraulic jump to back up onto the approach embankment.
4. Oblique hydraulic jump / bow-wave rise at a pier nose in supercritical
   approach flow (Ippen & Dawson theory, self-derived here from mass +
   tangential-velocity conservation across a stationary oblique wave,
   solved numerically) -- governs freeboard/afflux at piers sited in
   torrential reaches, distinct from ordinary (normal) jumps.
5. HEC-18 (CSU) local pier-scour depth and HEC-18 live-bed contraction
   scour -- standard empirical formulas used by transportation
   hydraulics practice (FHWA HEC-18), and the baseline that
   models/scour_ml.py benchmarks an ML model against.
6. Riprap (rock armor) sizing for pier protection.

References
----------
- Ippen, A.T. & Dawson, J.H. (1951), "Design of Channel Contractions",
  Trans. ASCE 116 -- oblique standing-wave theory for supercritical
  flow past a wall/pier deflection (see oblique_jump_* functions below;
  the theta-beta-Fr relation is re-derived here from first principles
  -- conservation of mass and of the tangential velocity component
  across a stationary oblique discontinuity -- rather than quoted, to
  keep the coefficients verifiable).
- Chow, V.T. (1959), "Open-Channel Hydraulics" -- Emin = 1.5*yc minimum
  specific energy result used in the choking check.
- FHWA HEC-18, 5th ed. -- Eq. 6.1 (pier scour) and Eq. 6.2 (live-bed
  contraction scour, y2 = y1*(Q2/Q1)^0.857*(W1/W2)^k1, k1 by V*/omega).
- FHWA HEC-23 -- Isbash-type riprap sizing at piers.
"""

from typing import Optional

import numpy as np
from scipy.optimize import brentq

from engines.sediment import settling_velocity

G = 9.81


# ---------------------------------------------------------------------------
# Flow regime, critical depth, specific energy, normal depth
# ---------------------------------------------------------------------------

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
    """Critical depth for a unit-width discharge q (m^2/s): yc = (q^2/g)^(1/3)."""
    return (q_unit_width ** 2 / g) ** (1 / 3)


def specific_energy(depth_m, velocity_mps, g=G):
    return depth_m + velocity_mps ** 2 / (2 * g)


def normal_depth_manning(Q, width_m, slope, manning_n, y_bounds=(1e-4, 200.0)):
    """
    Normal depth for a rectangular channel via Manning's equation,
    solved numerically (Manning's equation has no closed form in y):

        Q = (1/n) * A * R^(2/3) * sqrt(S),   A = b*y,  R = A / (b + 2y)

    Needed to classify a reach's slope type (steep/mild/critical) --
    the upper-course "torrential" designation applies to reaches whose
    *normal* depth is already below critical depth, not just to a
    single high-flow event.
    """
    def residual(y):
        A = width_m * y
        P = width_m + 2 * y
        R = A / P
        return (1.0 / manning_n) * A * R ** (2 / 3) * np.sqrt(slope) - Q

    return brentq(residual, y_bounds[0], y_bounds[1])


def slope_type(normal_depth_m, critical_depth_m) -> str:
    """
    Classify the channel slope type by comparing normal depth to
    critical depth (Chow 1959 classification):
      - normal < critical -> steep (S) slope: normal flow is itself
        supercritical -- the defining condition of a "torrential"
        upper-course reach.
      - normal > critical -> mild (M) slope: normal flow subcritical.
      - normal == critical -> critical (C) slope.
    """
    if normal_depth_m < 0.98 * critical_depth_m:
        return "steep (S) -- supercritical normal flow (torrential reach)"
    if normal_depth_m > 1.02 * critical_depth_m:
        return "mild (M) -- subcritical normal flow"
    return "critical (C) -- normal flow at critical depth"


# ---------------------------------------------------------------------------
# Ordinary (normal) hydraulic jump
# ---------------------------------------------------------------------------

def hydraulic_jump_conjugate_depth(h1, fr1):
    """
    Conjugate (sequent) depth after a normal hydraulic jump, from the
    Belanger equation -- used to size stilling basins where a
    torrential approach flow decelerates head-on into a barrage pool
    or bridge waterway.
    """
    return 0.5 * h1 * (np.sqrt(1 + 8 * fr1 ** 2) - 1)


def jump_length_estimate(h1, h2):
    """
    USBR simplified stilling-basin length estimate for a free
    hydraulic jump: L_j ~ 6*(h2 - h1). Use for a first-pass basin
    length; refine with USBR Type II/III design charts for final design.
    """
    return 6.0 * (h2 - h1)


# ---------------------------------------------------------------------------
# Bridge-opening choking check (supercritical approach)
# ---------------------------------------------------------------------------

def bridge_contraction_choking_check(Q, approach_width_m, opening_width_m,
                                      approach_depth_m, approach_velocity_mps) -> dict:
    """
    Checks whether a supercritical approach flow can pass through a
    narrower bridge opening without choking.

    A contraction can only pass a given discharge Q at or above the
    minimum specific energy for that unit discharge, Emin = 1.5*yc
    (Chow 1959). If the specific energy actually available at the
    approach section is less than Emin at the throat, the contraction
    "chokes": supercritical flow cannot get through, and a hydraulic
    jump forms and backs up onto the approach embankment -- a classic
    failure mode for bridges sited in torrential upper-course reaches
    with an undersized waterway.

    Returns
    -------
    dict with the throat critical depth/min. specific energy, the
    available approach specific energy, and a boolean `chokes`.
    """
    q_throat = Q / opening_width_m           # unit discharge at the throat
    yc_throat = critical_depth_rectangular(q_throat)
    Emin_throat = 1.5 * yc_throat

    E1 = specific_energy(approach_depth_m, approach_velocity_mps)

    return {
        "unit_discharge_throat_m2s": float(q_throat),
        "critical_depth_throat_m": float(yc_throat),
        "min_specific_energy_throat_m": float(Emin_throat),
        "available_specific_energy_m": float(E1),
        "chokes": bool(E1 < Emin_throat),
        "contraction_ratio": float(opening_width_m / approach_width_m),
    }


# ---------------------------------------------------------------------------
# Oblique hydraulic jump / pier bow-wave (Ippen & Dawson theory)
# ---------------------------------------------------------------------------
#
# Derivation: a stationary oblique wave front at angle beta to the
# upstream flow direction. Decompose the approach velocity into a
# component normal to the wave (Vn1 = V1 sin(beta)) and tangential to
# it (Vt1 = V1 cos(beta)). The normal component undergoes an ordinary
# hydraulic jump (mass + momentum normal to the front); the tangential
# component is unchanged (no force acts along the front). From those
# two facts:
#
#   Fr1n = Fr1 * sin(beta)
#   y2/y1 = 0.5 * (sqrt(1 + 8*Fr1n^2) - 1)                    [normal jump]
#   Vn2 = Vn1 * y1/y2                                          [continuity]
#   tan(beta - theta) = Vn2 / Vt2 = (y1/y2) * tan(beta)        [geometry]
#
# giving theta(beta, Fr1) = beta - atan[(y1/y2) * tan(beta)].
#
# Sanity checks: at beta = Mach angle (sin(beta) = 1/Fr1, so Fr1n = 1,
# y2 = y1) the wave has zero strength and theta -> 0, as expected for
# the weakest possible disturbance. At beta = 90 deg (a normal jump)
# theta -> 0 as well, since a head-on jump has no flow deflection.
# Between those limits theta(beta) rises to a maximum theta_max(Fr1);
# deflection angles below that maximum have two solutions (weak/
# attached and strong/detached), of which the weak (lower-beta)
# solution is the physically relevant one for an attached bow wave off
# a streamlined pier nose.

def _oblique_jump_depth_ratio(Fr1, beta_rad):
    Fr1n = Fr1 * np.sin(beta_rad)
    return 0.5 * (np.sqrt(1 + 8 * Fr1n ** 2) - 1)


def _oblique_jump_deflection(Fr1, beta_rad):
    y2_over_y1 = _oblique_jump_depth_ratio(Fr1, beta_rad)
    return beta_rad - np.arctan((1.0 / y2_over_y1) * np.tan(beta_rad))


def solve_wave_angle(Fr1, theta_deg) -> Optional[float]:
    """
    Solve for the (weak/attached) oblique-wave angle beta given the
    approach Froude number and the pier-nose deflection (half-)angle
    theta. Returns beta in degrees, or None if theta exceeds the
    maximum attainable deflection for this Fr1 (physically: the wave
    detaches into a curved bow shock ahead of the pier rather than
    staying attached at the nose -- see pier_bow_wave_rise's fallback).
    """
    theta_rad = np.radians(theta_deg)
    mach_angle = np.arcsin(1.0 / Fr1)

    betas = np.linspace(mach_angle + 1e-6, np.pi / 2 - 1e-6, 400)
    thetas = np.array([_oblique_jump_deflection(Fr1, b) for b in betas])
    peak_idx = int(np.argmax(thetas))
    theta_max = thetas[peak_idx]

    if theta_rad > theta_max:
        return None  # detached bow wave -- no attached weak-oblique-jump solution

    # weak-wave (attached) branch is between the Mach angle and the peak
    f = lambda b: _oblique_jump_deflection(Fr1, b) - theta_rad
    beta = brentq(f, mach_angle + 1e-6, betas[peak_idx])
    return float(np.degrees(beta))


def pier_bow_wave_rise(Fr1, y1, nose_half_angle_deg=20.0, blunt_nose_factor=0.5) -> dict:
    """
    Bow-wave (standing-wave) rise at a bridge-pier nose under
    supercritical approach flow -- the freeboard-governing surge that
    an ordinary (head-on) hydraulic jump calculation misses, since the
    wave stands obliquely off the nose rather than perpendicular to
    the flow.

    nose_half_angle_deg : effective wedge half-angle of the pier
        cutwater (~10-15 deg for a sharp pointed nose, ~20-30 deg as a
        rough equivalent for a rounded/semicircular nose -- confirm
        against the actual pier drawing; this is a shape idealization).
    blunt_nose_factor : fallback rise coefficient (rise = factor *
        Fr1^2 * y1) used only if the requested nose angle exceeds the
        maximum attached-wave deflection for this Fr1 (bow wave
        detaches -- treat as a blunt-body approximation instead).

    Returns
    -------
    dict with wave_angle_deg (None if detached), depth_ratio y2/y1,
    rise_m, and a `detached` flag.
    """
    beta_deg = solve_wave_angle(Fr1, nose_half_angle_deg)

    if beta_deg is None:
        rise = blunt_nose_factor * Fr1 ** 2 * y1
        return {
            "detached": True,
            "wave_angle_deg": None,
            "depth_ratio": float(1 + rise / y1),
            "rise_m": float(rise),
            "note": "Requested nose angle exceeds the attached-wave limit for this "
                    "Froude number -- bow wave detaches into a curved shock ahead of "
                    "the pier. Rise estimated via a blunt-body approximation "
                    "(rise = factor * Fr1^2 * y1); consider a sharper cutwater.",
        }

    y2_over_y1 = _oblique_jump_depth_ratio(Fr1, np.radians(beta_deg))
    rise = y1 * (y2_over_y1 - 1)
    return {
        "detached": False,
        "wave_angle_deg": beta_deg,
        "depth_ratio": float(y2_over_y1),
        "rise_m": float(rise),
        "note": None,
    }


# ---------------------------------------------------------------------------
# HEC-18 local pier scour
# ---------------------------------------------------------------------------

def hec18_pier_scour_depth(y1, V1, pier_width, K1=1.0, K2=1.0, K3=1.1, K4=1.0):
    """
    HEC-18 (CSU) equation for local live-bed scour depth at a bridge
    pier -- the standard empirical baseline used by transportation
    hydraulics practice (FHWA HEC-18 Eq. 6.1):

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


# ---------------------------------------------------------------------------
# HEC-18 live-bed contraction scour
# ---------------------------------------------------------------------------

def _contraction_k1(shear_vel_mps, settling_vel_mps) -> float:
    """
    HEC-18 exponent k1, selected by mode of bed-material transport
    (V*/omega ratio):
        < 0.50        -> 0.59  mostly contact bed-material discharge
        0.50 to 2.0   -> 0.64  some suspended bed-material discharge
        > 2.0         -> 0.69  mostly suspended bed-material discharge
    """
    ratio = shear_vel_mps / max(settling_vel_mps, 1e-9)
    if ratio < 0.50:
        return 0.59
    if ratio <= 2.0:
        return 0.64
    return 0.69


def hec18_contraction_scour_live_bed(Q1, Q2, W1, W2, y1, slope, d50_m,
                                      y0_contracted=None) -> dict:
    """
    HEC-18 live-bed contraction scour (Eq. 6.2):

        y2 = y1 * (Q2/Q1)^0.857 * (W1/W2)^k1

    Parameters
    ----------
    Q1, W1 : discharge (m^3/s) and bottom width (m) transporting bed
             material in the upstream approach section
    Q2, W2 : discharge and bottom width in the contracted (bridge
             opening) section, less pier widths
    y1 : average approach depth (m)
    slope : approach energy/bed slope, for shear velocity V* = sqrt(g y1 S)
    d50_m : bed-material median grain size, for fall velocity (mode-of-
            transport k1 selection) via engines.sediment.settling_velocity
    y0_contracted : existing depth in the contracted section before
                    scour (m); if omitted, y1 is used per HEC-18's own
                    simplifying assumption that approach and pre-scour
                    contracted depths are equal.

    Returns
    -------
    dict with y2 (post-scour depth in the contraction), the contraction
    scour depth ys = y2 - y0, and the k1 exponent used.
    """
    v_star = np.sqrt(G * y1 * slope)
    omega = settling_velocity(d50_m)
    k1 = _contraction_k1(v_star, omega)

    y2 = y1 * (Q2 / Q1) ** 0.857 * (W1 / W2) ** k1
    y0 = y1 if y0_contracted is None else y0_contracted
    ys = y2 - y0

    return {
        "shear_velocity_mps": float(v_star),
        "fall_velocity_mps": float(omega),
        "k1_exponent": k1,
        "contracted_depth_y2_m": float(y2),
        "contraction_scour_depth_m": float(max(ys, 0.0)),
    }


# ---------------------------------------------------------------------------
# Riprap sizing and combined pier-protection design
# ---------------------------------------------------------------------------

def riprap_d50_for_pier(V1, specific_gravity=2.65, safety_factor=1.0):
    """
    Riprap median stone size for pier scour protection, per the
    Isbash-type relation commonly used in FHWA HEC-23:
        d50 = 0.692 * V1^2 / [ (Ss - 1) * 2g ]  (SI, at-pier sizing)
    """
    d50 = 0.692 * V1 ** 2 / ((specific_gravity - 1) * 2 * G)
    return float(d50 * safety_factor)


def pier_protection_design(y1, V1, pier_width, nose_half_angle_deg=None,
                            contraction: Optional[dict] = None, **hec18_kwargs) -> dict:
    """
    Combined scour + riprap protection summary for a single pier.

    nose_half_angle_deg : if given, also computes the oblique bow-wave
        rise at the pier nose (only meaningful when the approach flow
        is supercritical -- ignored otherwise).
    contraction : optional dict of kwargs to also run
        hec18_contraction_scour_live_bed() and fold its scour into the
        footing-depth recommendation alongside local pier scour (the
        two scour components are additive per HEC-18 practice).
    """
    regime = flow_regime(V1, y1)
    fr1 = float(froude_number(V1, y1))
    ys_local = hec18_pier_scour_depth(y1, V1, pier_width, **hec18_kwargs)
    d50_riprap = riprap_d50_for_pier(V1)

    result = {
        "flow_regime": regime,
        "froude_number": fr1,
        "predicted_local_scour_depth_m": ys_local,
        "recommended_riprap_d50_m": d50_riprap,
    }

    ys_contraction = 0.0
    if contraction is not None:
        c = hec18_contraction_scour_live_bed(**contraction)
        result["contraction_scour"] = c
        ys_contraction = c["contraction_scour_depth_m"]

    if nose_half_angle_deg is not None and fr1 > 1.0:
        result["bow_wave"] = pier_bow_wave_rise(fr1, y1, nose_half_angle_deg)

    total_scour = ys_local + ys_contraction
    result["total_scour_depth_m"] = total_scour
    result["recommended_footing_depth_m"] = total_scour * 1.3  # 30% freeboard-on-scour practice
    return result