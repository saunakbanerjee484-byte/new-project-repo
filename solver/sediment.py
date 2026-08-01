"""
engines/sediment.py

Sediment Transport: Shields Parameter, Critical Shear Stress & Bedload/
Suspended-Load Assessment.

Feeds three consumers in the platform:
  - general channel-stability / aggradation-degradation assessment
  - engines/bridge_torrents.py (pier-scour bed-condition context)
  - models/scour_ml.py (feature engineering for the scour ML model)

Covers, in the order a channel-stability check typically uses them:
  1. Bed shear stress (normal-flow approximation)
  2. Dimensionless grain size D* and the Shields (1936) mobility
     parameter theta, compared against the Soulsby (1997) explicit fit
     for the critical Shields number theta_cr -- avoids an iterative
     Shields-diagram lookup.
  3. Settling velocity (Soulsby 1997 explicit formula) and the Rouse
     number, which classifies transport mode: bed load, mixed
     suspended+bed load, or wash load.
  4. Meyer-Peter & Muller (1948) bedload transport rate once motion is
     confirmed.

References
----------
- Shields, A. (1936), "Anwendung der Aehnlichkeitsmechanik...".
- Soulsby, R.L. (1997), "Dynamics of Marine Sands", Thomas Telford --
  source of both the D*-based critical Shields fit and the explicit
  settling-velocity formula used below.
- Meyer-Peter, E. & Muller, R. (1948), "Formulas for Bed-Load Transport",
  IAHSR 2nd meeting -- qb* = 8*(theta - theta_cr)^1.5.
- Rouse, H. (1937) -- suspension criterion via ws / (kappa * u*).
"""

from dataclasses import dataclass

import numpy as np

G = 9.81
RHO_WATER = 1000.0
RHO_SEDIMENT = 2650.0   # quartz sand/gravel, typical alluvial value
KAPPA = 0.40             # von Karman constant
NU_WATER = 1.0e-6        # kinematic viscosity of water, m^2/s, ~20 C


# ---------------------------------------------------------------------------
# Shear stress & Shields mobility parameter
# ---------------------------------------------------------------------------

def bed_shear_stress(depth_m, slope, rho_w=RHO_WATER):
    """tau = rho * g * h * S (normal-flow / wide-channel approximation)."""
    return rho_w * G * depth_m * slope


def shear_velocity(tau_bed_pa, rho_w=RHO_WATER):
    """u* = sqrt(tau / rho) -- the characteristic near-bed turbulent velocity scale."""
    return np.sqrt(np.maximum(tau_bed_pa, 0.0) / rho_w)


def dimensionless_grain_size(d50_m, rho_s=RHO_SEDIMENT, rho_w=RHO_WATER, nu=NU_WATER):
    """D* = d50 * [(rho_s/rho_w - 1) * g / nu^2] ^ (1/3) -- Bonnefille (1963)."""
    return d50_m * ((rho_s / rho_w - 1) * G / nu ** 2) ** (1 / 3)


def shields_parameter(tau_bed_pa, d50_m, rho_s=RHO_SEDIMENT, rho_w=RHO_WATER):
    """theta = tau / [(rho_s - rho_w) * g * d50] -- ratio of destabilizing drag to submerged grain weight."""
    return tau_bed_pa / ((rho_s - rho_w) * G * d50_m)


def critical_shields_number(d_star) -> float:
    """
    Soulsby (1997) explicit fit for the critical Shields number as a
    function of dimensionless grain size D*, reproducing the classic
    Shields-diagram curve without iteration:

        theta_cr = 0.30 / (1 + 1.2*D*) + 0.055 * [1 - exp(-0.020*D*)]
    """
    return 0.30 / (1 + 1.2 * d_star) + 0.055 * (1 - np.exp(-0.020 * d_star))


def is_bed_mobile(depth_m, slope, d50_m, rho_s=RHO_SEDIMENT, rho_w=RHO_WATER) -> dict:
    """Compare actual vs. critical Shields parameter for incipient motion of the bed material."""
    tau = bed_shear_stress(depth_m, slope, rho_w)
    theta = shields_parameter(tau, d50_m, rho_s, rho_w)
    d_star = dimensionless_grain_size(d50_m, rho_s, rho_w)
    theta_cr = critical_shields_number(d_star)
    return {
        "tau_bed_pa": float(tau),
        "shear_velocity_mps": float(shear_velocity(tau, rho_w)),
        "dimensionless_grain_size": float(d_star),
        "shields_parameter": float(theta),
        "critical_shields_number": float(theta_cr),
        "mobile": bool(theta > theta_cr),
        "mobility_ratio": float(theta / theta_cr),
    }


# ---------------------------------------------------------------------------
# Settling velocity & suspension criterion (Rouse number)
# ---------------------------------------------------------------------------

def settling_velocity(d50_m, rho_s=RHO_SEDIMENT, rho_w=RHO_WATER, nu=NU_WATER):
    """
    Soulsby (1997) explicit formula for grain settling (fall) velocity,
    valid across the full range from Stokes (fine silt) to Newtonian
    (coarse gravel) drag regimes without needing a drag-coefficient
    iteration:

        ws = (nu / d50) * [ sqrt(10.36^2 + 1.049*D*^3) - 10.36 ]
    """
    d_star = dimensionless_grain_size(d50_m, rho_s, rho_w, nu)
    return (nu / d50_m) * (np.sqrt(10.36 ** 2 + 1.049 * d_star ** 3) - 10.36)


def rouse_number(settling_velocity_mps, shear_velocity_mps, kappa=KAPPA):
    """
    Z = ws / (kappa * u*) -- governs whether mobilized sediment travels
    as bed load, mixed load, or wash load (suspended indefinitely).
    """
    return settling_velocity_mps / (kappa * max(shear_velocity_mps, 1e-9))


def classify_transport_mode(rouse_no: float) -> str:
    """Standard Rouse-number transport-mode bands (van Rijn 1984 / ASCE Manual 110)."""
    if rouse_no > 2.5:
        return "bed_load"
    if rouse_no > 1.2:
        return "mixed_load (50% suspended)"
    if rouse_no > 0.8:
        return "mixed_load (100% suspended)"
    return "wash_load (fully suspended, negligible settling)"


# ---------------------------------------------------------------------------
# Bedload transport rate (Meyer-Peter & Muller, 1948)
# ---------------------------------------------------------------------------

def meyer_peter_muller_bedload(theta, theta_cr, d50_m, rho_s=RHO_SEDIMENT, rho_w=RHO_WATER):
    """
    Meyer-Peter & Muller (1948) volumetric bedload transport rate per
    unit channel width.

        qb* = 8 * (theta - theta_cr)^1.5                    (dimensionless)
        qb  = qb* * sqrt[(rho_s/rho_w - 1) * g * d50^3]      (m^2/s, i.e. m^3/s per m width)

    Returns 0 for theta <= theta_cr (no transport).
    """
    excess = max(theta - theta_cr, 0.0)
    qb_star = 8.0 * excess ** 1.5
    qb = qb_star * np.sqrt((rho_s / rho_w - 1) * G * d50_m ** 3)
    return float(qb)


# ---------------------------------------------------------------------------
# Combined channel-reach sediment assessment
# ---------------------------------------------------------------------------

@dataclass
class SedimentAssessment:
    tau_bed_pa: float
    shear_velocity_mps: float
    shields_parameter: float
    critical_shields_number: float
    mobile: bool
    mobility_ratio: float
    settling_velocity_mps: float
    rouse_number: float
    transport_mode: str
    bedload_rate_m2_per_s: float
    bedload_rate_tonnes_per_day_per_m: float


def assess_reach(depth_m, slope, d50_m, width_m=None,
                  rho_s=RHO_SEDIMENT, rho_w=RHO_WATER, nu=NU_WATER) -> SedimentAssessment:
    """
    One-call sediment-transport assessment for a channel cross-section --
    the combined entry point most callers (dashboards, scour_ml feature
    prep, channel-stability checks) should use instead of chaining the
    individual functions above by hand.

    Parameters
    ----------
    depth_m, slope : normal-flow depth (m) and bed/energy slope (-)
    d50_m : median bed-material grain size (m)
    width_m : optional channel width (m) -- if given, also reports the
              total reach bedload rate (not just per unit width).
    """
    tau = bed_shear_stress(depth_m, slope, rho_w)
    u_star = shear_velocity(tau, rho_w)
    d_star = dimensionless_grain_size(d50_m, rho_s, rho_w, nu)
    theta = shields_parameter(tau, d50_m, rho_s, rho_w)
    theta_cr = critical_shields_number(d_star)
    mobile = theta > theta_cr

    ws = settling_velocity(d50_m, rho_s, rho_w, nu)
    rouse = rouse_number(ws, u_star)
    mode = classify_transport_mode(rouse)

    qb = meyer_peter_muller_bedload(theta, theta_cr, d50_m, rho_s, rho_w) if mobile else 0.0
    # convert m^3/s per m width -> tonnes/day per m width, using dry bulk unit weight of sediment
    qb_tpd_per_m = qb * rho_s * 86400 / 1000.0

    return SedimentAssessment(
        tau_bed_pa=float(tau),
        shear_velocity_mps=float(u_star),
        shields_parameter=float(theta),
        critical_shields_number=float(theta_cr),
        mobile=bool(mobile),
        mobility_ratio=float(theta / theta_cr),
        settling_velocity_mps=float(ws),
        rouse_number=float(rouse),
        transport_mode=mode,
        bedload_rate_m2_per_s=float(qb),
        bedload_rate_tonnes_per_day_per_m=float(qb_tpd_per_m),
    )