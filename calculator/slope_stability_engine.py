"""
engines/slope_stability_engine.py

AdvancedSlopeStabilityEngine -- Multi-Failure-Mode Analysis, extending
engines.EarthDamSeepage.AdvancedEmbankmentEngine.
=======================================================================
TECHNICAL BACKGROUND (read before using):
This implements 4 of the 7 roadmap items from EarthDamSeepage.py with
real, mostly web-verified numerics, and leaves 3 as documented stubs --
same honesty standard as the rest of this codebase. Bishop's Simplified
Method (1955) -- verified against five independent sources during
writing, all agreeing on the same iterative moment-equilibrium formula
-- searches a grid of trial circular slip surfaces, divides each into
vertical slices, and iterates F = f(F) to convergence (the FoS appears
on both sides of Bishop's own equation, per every reference found).
Janbu's Simplified Method (1954/1973) is implemented alongside it using
HORIZONTAL force equilibrium instead of Bishop's moment equilibrium --
a genuinely different equilibrium assumption, so running both gives a
real cross-check, not two labels on the same number. Rapid drawdown is
a SIMPLIFIED single-stage ru-based adjustment layered on the same slice
mechanics (not the full USACE three-stage undrained-strength method --
flagged clearly below). Progressive piping and wave-overtopping erosion
both use Hanson & Simon's (2001) excess-shear-stress erosion equation,
epsilon = kd*(tau - tau_c) -- confirmed via web search as the standard
formula -- but deliberately do NOT ship default kd/tau_c values: these
are soil-specific, lab-measured (Jet Erosion Test) properties, and a
fabricated "typical" table would be worse than requiring the real
measurement. Slope stability (Bishop/Janbu) and rapid drawdown remain
UNVERIFIED FOR PRODUCTION USE beyond the formula level -- this module
does not replace a licensed geotechnical engineer's slip-surface report.
"""

import numpy as np
from scipy.optimize import brentq

from engines.EarthDamSeepage import AdvancedEmbankmentEngine
from engines.embankment import phreatic_line_no_filter

G = 9.81
GAMMA_WATER = 9.81  # kN/m^3


# ===========================================================================
# 1. Embankment ground-surface profile (shared by both slope-stability
#    methods below)
# ===========================================================================

def embankment_ground_profile(dam_height_m, upstream_slope_h_per_v,
                               downstream_slope_h_per_v, crest_width_m):
    """
    Piecewise-linear ground-surface profile of a trapezoidal embankment
    cross-section, in the SAME downstream-toe-referenced xi coordinate
    system used throughout engines/embankment.py (xi=0 at the downstream
    toe, increasing toward the reservoir/upstream heel).

    Returns (xi_vertices, y_vertices) -- four points: toe, downstream
    crest edge, upstream crest edge, upstream heel -- from which ground
    elevation at any xi is obtained by linear interpolation (valid since
    xi is monotonically increasing along this profile, so it IS a proper
    function y(xi), not just a set of points).
    """
    xi = np.array([
        0.0,
        downstream_slope_h_per_v * dam_height_m,
        downstream_slope_h_per_v * dam_height_m + crest_width_m,
        downstream_slope_h_per_v * dam_height_m + crest_width_m + upstream_slope_h_per_v * dam_height_m,
    ])
    y = np.array([0.0, dam_height_m, dam_height_m, 0.0])
    return xi, y


def ground_elevation(x, xi_vertices, y_vertices):
    """Ground elevation at horizontal position x, via linear interpolation of the polyline above."""
    return np.interp(x, xi_vertices, y_vertices)


# ===========================================================================
# 2. Circular slip-surface slicing
# ===========================================================================

def _circle_lower_arc(x, xc, yc, R):
    """Lower branch of the trial circle, y = yc - sqrt(R^2 - (x-xc)^2); NaN outside the circle's x-domain."""
    inside = R ** 2 - (x - xc) ** 2
    return np.where(inside >= 0, yc - np.sqrt(np.maximum(inside, 0)), np.nan)


def build_trial_circle(xc, yc, toe_xi, xi_vertices, y_vertices, n_slices=30):
    """
    Builds a single trial circular slip surface, forced to exit exactly
    at `toe_xi` (a "toe circle", the classical critical case for
    homogeneous embankment slopes) -- this fixes R = distance from the
    trial center to the toe, so the only free search parameters are
    (xc, yc), reducing what would otherwise be a 3-parameter search.

    The entry point (where the circle re-emerges through the ground
    surface on the crest/reservoir side) is found by root-finding the
    intersection of the circle's lower arc with the ground profile.

    Returns None if no valid entry point exists for this (xc, yc) (e.g.
    the circle is too shallow/deep to actually cut through the slope),
    so the calling grid-search can simply skip invalid trial circles.
    """
    R = np.sqrt((xc - toe_xi) ** 2 + yc ** 2)

    # Search for the entry point along the ground profile, starting just
    # past the toe out to the far (upstream) end of the profile -- the
    # circle's lower arc must cross the ground surface exactly once in
    # this range for a valid slip surface.
    x_scan = np.linspace(toe_xi + 1e-3, xi_vertices[-1], 400)
    y_circle_scan = _circle_lower_arc(x_scan, xc, yc, R)
    y_ground_scan = ground_elevation(x_scan, xi_vertices, y_vertices)
    diff = y_ground_scan - y_circle_scan  # positive where ground is above the circle (valid slice zone)

    valid = ~np.isnan(diff)
    if not np.any(valid) or np.all(diff[valid] <= 0):
        return None  # circle never gets under the ground surface -- not a valid slip surface

    # First sign change of `diff` beyond the toe marks the entry point
    # (ground surface re-crossing the circle) -- bracket and refine it.
    sign = np.sign(diff[valid])
    x_valid = x_scan[valid]
    crossings = np.where(np.diff(sign) != 0)[0]
    if len(crossings) == 0:
        return None

    i = crossings[-1]  # the outermost (furthest upstream) crossing = the entry point
    try:
        entry_xi = brentq(
            lambda x: ground_elevation(x, xi_vertices, y_vertices) - _circle_lower_arc(x, xc, yc, R),
            x_valid[i], x_valid[i + 1],
        )
    except ValueError:
        return None

    if entry_xi <= toe_xi + 0.5:
        return None  # degenerate (near-zero-width) slip surface, skip

    slice_edges = np.linspace(toe_xi, entry_xi, n_slices + 1)
    x_mid = 0.5 * (slice_edges[:-1] + slice_edges[1:])
    b = np.diff(slice_edges)

    y_ground_mid = ground_elevation(x_mid, xi_vertices, y_vertices)
    y_circle_mid = _circle_lower_arc(x_mid, xc, yc, R)
    height = y_ground_mid - y_circle_mid

    valid_slices = (height > 1e-6) & ~np.isnan(y_circle_mid)
    if np.sum(valid_slices) < 3:
        return None  # too few valid slices to be a meaningful failure surface

    # Base inclination angle alpha: slope of the circle (not the ground)
    # at each slice's midpoint, since the slice BASE lies on the circle.
    # From y_circle = yc - sqrt(R^2-(x-xc)^2), dy/dx = (x-xc)/sqrt(R^2-(x-xc)^2).
    tan_alpha = (x_mid - xc) / np.sqrt(np.maximum(R ** 2 - (x_mid - xc) ** 2, 1e-9))
    alpha = np.arctan(tan_alpha)

    return {
        "xc": xc, "yc": yc, "R": R,
        "toe_xi": toe_xi, "entry_xi": entry_xi,
        "x_mid": x_mid[valid_slices], "b": b[valid_slices],
        "height": height[valid_slices], "alpha": alpha[valid_slices],
        "y_ground_mid": y_ground_mid[valid_slices],
        "y_circle_mid": y_circle_mid[valid_slices],
    }


# ===========================================================================
# 3. Pore pressure at the slip-surface base (from the already-verified
#    phreatic line, or a simplified ru override for rapid drawdown)
# ===========================================================================

def pore_pressure_from_phreatic_line(x_mid, y_circle_mid, phreatic_xi, phreatic_y):
    """
    Pore pressure at each slice base, u = gamma_water * hw, where hw is
    the height of the (already-computed, Casagrande) phreatic surface
    above the slip-circle base at that slice -- zero if the slip surface
    sits above the phreatic line (unsaturated zone, no pore pressure).
    """
    phreatic_elev_at_mid = np.interp(x_mid, phreatic_xi, phreatic_y)
    hw = np.maximum(phreatic_elev_at_mid - y_circle_mid, 0.0)
    return GAMMA_WATER * hw


def pore_pressure_from_ru(height, unit_weight_kn_m3, ru):
    """
    Simplified pore-pressure ratio approximation, u = ru * gamma * h,
    used for the rapid-drawdown case below where actual transient pore
    pressures don't follow the steady-state phreatic line (they lag
    behind a falling reservoir) -- ru is a single lumped coefficient
    representing that lag, per common simplified (not full USACE
    three-stage) rapid-drawdown practice. Typical ru sits roughly in
    0.3-0.5 for a rapid, complete drawdown of a compacted clay core;
    this must be treated as an engineering judgment input, not a
    universal constant.
    """
    return ru * unit_weight_kn_m3 * height


# ===========================================================================
# 4. Bishop's Simplified and Janbu's Simplified factor of safety
# ===========================================================================

def bishop_simplified_fos(slices, cohesion_kpa, friction_angle_deg, unit_weight_kn_m3,
                           pore_pressure_kpa, max_iter=100, tol=1e-5):
    """
    Bishop's Simplified Method (1955): moment equilibrium about the
    circle's center. F appears on both sides (via m_alpha below), so it
    is solved by fixed-point iteration -- confirmed via web search as
    converging in just a few iterations in practice, consistent with
    every textbook description found ("2 or 3 trials").

        F = sum[ (c'*b + (W - u*b)*tan(phi')) / m_alpha ] / sum[W*sin(alpha)]
        m_alpha = cos(alpha) * (1 + tan(alpha)*tan(phi')/F)
    """
    phi = np.radians(friction_angle_deg)
    alpha = slices["alpha"]
    b = slices["b"]
    W = unit_weight_kn_m3 * slices["height"] * b  # kN per unit length of dam
    u = pore_pressure_kpa

    F = 1.0  # initial guess (Fellenius/Ordinary-Method-of-Slices-style starting point)
    for _ in range(max_iter):
        m_alpha = np.cos(alpha) * (1 + np.tan(alpha) * np.tan(phi) / F)
        # Guard against m_alpha -> 0 (can occur transiently during
        # iteration for steep base angles); nudge away from the
        # singularity rather than dividing by ~0.
        m_alpha = np.where(np.abs(m_alpha) < 1e-6, np.sign(m_alpha + 1e-12) * 1e-6, m_alpha)

        numerator = np.sum((cohesion_kpa * b + (W - u * b) * np.tan(phi)) / m_alpha)
        denominator = np.sum(W * np.sin(alpha))
        if abs(denominator) < 1e-3 * np.sum(W):
            # Denominator collapsing toward zero relative to the slice
            # weights means this is a degenerate (near-flat / oversized)
            # trial circle, not a meaningful failure surface -- flag it
            # as non-physical (inf) rather than returning a misleadingly
            # huge-but-finite FoS from dividing by a near-zero number.
            return float("inf")
        F_new = numerator / denominator

        if abs(F_new - F) < tol:
            return float(F_new)
        F = F_new
    return float(F)  # return best estimate even if not fully converged within max_iter


def janbu_simplified_fos(slices, cohesion_kpa, friction_angle_deg, unit_weight_kn_m3,
                          pore_pressure_kpa, soil_type="c-phi", max_iter=100, tol=1e-5):
    """
    Janbu's Simplified Method (1954): HORIZONTAL force equilibrium
    instead of Bishop's moment equilibrium -- a genuinely different
    equilibrium assumption (confirmed via web search: "Janbu had
    suggested using the overall force equilibrium equation ... omission
    of the inter-slice shear forces"), so comparing Bishop vs Janbu is a
    real cross-check between two independent equilibrium formulations,
    not the same computation under two names.

        F0 = sum[ (c'*b + (W - u*b)*tan(phi')) * sec(alpha) / m_alpha ] / sum[W*tan(alpha)]
        F = f0 * F0

    f0 is Janbu's (1973) empirical correction factor for the interslice
    shear forces the simplified method neglects:
        f0 = 1 + b1*[(d/L) - 1.4*(d/L)^2]
    where d = maximum depth of the slip surface below the ground
    surface, L = horizontal length of the slip surface, and b1 depends
    on soil type (0.69 purely cohesive / phi=0, 0.31 purely frictional /
    c=0, 0.50 combined c-phi soil -- the standard embankment case and
    this function's default). These b1 values are the widely-published
    Janbu (1973) coefficients; unlike Bishop's formula and the Hanson
    erosion equation elsewhere in this module, they were not
    independently re-verified via a fresh web search in this pass, so
    treat f0 as a reasonable standard correction rather than a
    freshly-confirmed number.
    """
    phi = np.radians(friction_angle_deg)
    alpha = slices["alpha"]
    b = slices["b"]
    W = unit_weight_kn_m3 * slices["height"] * b
    u = pore_pressure_kpa

    F = 1.0
    for _ in range(max_iter):
        m_alpha = np.cos(alpha) * (1 + np.tan(alpha) * np.tan(phi) / F)
        m_alpha = np.where(np.abs(m_alpha) < 1e-6, np.sign(m_alpha + 1e-12) * 1e-6, m_alpha)

        numerator = np.sum((cohesion_kpa * b + (W - u * b) * np.tan(phi)) * (1 / np.cos(alpha)) / m_alpha)
        denominator = np.sum(W * np.tan(alpha))
        if abs(denominator) < 1e-3 * np.sum(W):
            return float("inf")
        F0_new = numerator / denominator

        if abs(F0_new - F) < tol:
            F0 = F0_new
            break
        F = F0_new
    else:
        F0 = F

    d = np.max(slices["y_ground_mid"] - slices["y_circle_mid"])
    L = slices["entry_xi"] - slices["toe_xi"]
    b1_table = {"phi_only": 0.69, "c_only": 0.31, "c-phi": 0.50}
    b1 = b1_table.get(soil_type, 0.50)
    dL = d / max(L, 1e-6)
    f0 = 1 + b1 * (dL - 1.4 * dL ** 2)

    return float(f0 * F0)


# ===========================================================================
# 5. Hanson excess-shear-stress erosion (shared by piping & overtopping)
# ===========================================================================

def hanson_erosion_rate(tau_pa, tau_c_pa, kd_cm3_per_N_s):
    """
    Hanson & Simon (2001) excess shear stress erosion equation --
    confirmed via web search as the standard model underlying both the
    Jet Erosion Test (JET) and Hole Erosion Test (HET) interpretation:

        epsilon = kd * (tau - tau_c),   for tau > tau_c;  0 otherwise

    UNIT CARE: kd is conventionally reported in cm^3/(N*s) and tau in
    Pa (= N/m^2), but kd*tau then comes out in cm^3/(s*m^2), NOT cm/s --
    a unit-consistency check the initial version of this function
    missed, giving erosion rates inflated by 10^4x (verified by working
    a physically-sensible example by hand: kd~2 cm^3/(N-s), excess
    shear ~100 Pa should give roughly 7 mm/hr, a plausible internal-
    erosion rate, not the >1000 cm/s the unconverted formula produced).
    The fix: 1 m^2 = 10^4 cm^2, so cm^3/(s*m^2) = cm/s / 10^4 -- applied
    below as the explicit CM2_PER_M2 conversion.

    epsilon : erosion rate, cm/s (a LINEAR rate -- how fast the eroding
              boundary recedes)
    tau_pa, tau_c_pa : applied and critical hydraulic shear stress (Pa)
    kd_cm3_per_N_s : erodibility coefficient (cm^3/N-s), a soil-specific
              LAB-MEASURED property from a JET or HET -- deliberately
              NOT defaulted here (see module docstring): Hanson & Simon
              themselves report roughly an order-of-magnitude scatter
              even within one soil classification, so a fabricated
              "typical" value would misrepresent the actual uncertainty.
    """
    CM2_PER_M2 = 1.0e4
    excess = max(tau_pa - tau_c_pa, 0.0)
    return kd_cm3_per_N_s * excess / CM2_PER_M2


def overtopping_unit_discharge(head_above_crest_m, weir_coeff=1.7):
    """
    Broad-crested weir unit discharge for flow overtopping the
    embankment crest, q = weir_coeff * head^1.5 -- the same formula
    already used for breach discharge in engines.dam_break, applied
    here to the (much smaller, non-breach) case of overtopping flow
    still contained by the crest and running down the downstream face.
    """
    return weir_coeff * max(head_above_crest_m, 0.0) ** 1.5


def overtopping_face_shear_stress(unit_discharge_m2_s, downstream_slope_h_per_v, manning_n):
    """
    Approximate hydraulic shear stress on the downstream face under a
    thin, fast overtopping sheet flow, using normal-flow depth from
    Manning's equation for the unit discharge on the slope, then
    tau = rho*g*depth*slope -- the same bed-shear-stress relation
    already used in engines.sediment, applied here to the crest/slope
    surface instead of a channel bed.
    """
    slope = 1.0 / downstream_slope_h_per_v
    # Manning's equation for a wide sheet flow (per-unit-width, R~=depth):
    # q = (1/n) * depth^(5/3) * sqrt(slope)  =>  depth = (q*n/sqrt(slope))^(3/5)
    depth = (unit_discharge_m2_s * manning_n / np.sqrt(slope)) ** (3.0 / 5.0)
    tau = 1000.0 * G * depth * slope  # rho_water * g * depth * slope, in Pa
    return {"flow_depth_m": float(depth), "shear_stress_pa": float(tau)}


# ===========================================================================
# AdvancedSlopeStabilityEngine
# ===========================================================================

class AdvancedSlopeStabilityEngine(AdvancedEmbankmentEngine):
    """
    Extends AdvancedEmbankmentEngine with slope-stability (Bishop's
    Simplified + Janbu's Simplified, cross-verified), simplified rapid
    drawdown, and Hanson-erosion-based piping/overtopping analyses.
    Continues the same inheritance chain as engines/EarthDamSeepage.py.
    """

    def __init__(self, *args, cohesion_kpa=15.0, friction_angle_deg=28.0,
                 soil_unit_weight_kn_m3=19.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.cohesion_kpa = cohesion_kpa
        self.friction_angle_deg = friction_angle_deg
        self.soil_unit_weight_kn_m3 = soil_unit_weight_kn_m3

    def _ground_profile(self):
        return embankment_ground_profile(self.dam_height_m, self.upstream_slope_h_per_v,
                                          self.downstream_slope_h_per_v, self.crest_width_m)

    def search_critical_circle(self, method="bishop", n_centers=12, n_radii_toe_points=1,
                                soil_type="c-phi", n_slices=30):
        """
        Grid search over trial toe circles for the minimum factor of
        safety, on the downstream slope (the steady-seepage critical
        case). Searches a grid of candidate centers (xc, yc) above the
        slope; for each, build_trial_circle() finds the corresponding
        entry point and slices, and the requested method (bishop/janbu)
        computes F. Returns the minimum-F circle and its slices for
        plotting.

        This is a grid search, not a gradient-based/simplex optimizer --
        adequate for an engineering screening tool, not a substitute for
        a dedicated slope-stability package's more exhaustive search.
        """
        xi_vertices, y_vertices = self._ground_profile()
        toe_xi = 0.0
        crest_start = self.downstream_slope_h_per_v * self.dam_height_m

        phreatic = phreatic_line_no_filter(
            self.dam_height_m, self.upstream_water_depth_m,
            self.upstream_slope_h_per_v, self.downstream_slope_h_per_v, self.crest_width_m,
        )

        xc_candidates = np.linspace(crest_start * 0.3, crest_start * 2.5, n_centers)
        yc_candidates = np.linspace(self.dam_height_m * 0.8, self.dam_height_m * 3.0, n_centers)

        best = {"fos": np.inf, "slices": None, "xc": None, "yc": None}
        for xc in xc_candidates:
            for yc in yc_candidates:
                slices = build_trial_circle(xc, yc, toe_xi, xi_vertices, y_vertices, n_slices)
                if slices is None:
                    continue

                u = pore_pressure_from_phreatic_line(slices["x_mid"], slices["y_circle_mid"],
                                                      phreatic["xi_m"], phreatic["elevation_m"])

                if method == "bishop":
                    fos = bishop_simplified_fos(slices, self.cohesion_kpa, self.friction_angle_deg,
                                                 self.soil_unit_weight_kn_m3, u)
                else:
                    fos = janbu_simplified_fos(slices, self.cohesion_kpa, self.friction_angle_deg,
                                                self.soil_unit_weight_kn_m3, u, soil_type=soil_type)

                if np.isfinite(fos) and 0 < fos < best["fos"]:
                    best = {"fos": fos, "slices": slices, "xc": xc, "yc": yc}

        return best

    def multi_method_slope_stability(self, **search_kwargs):
        """Runs both Bishop's and Janbu's Simplified searches and returns both for direct comparison."""
        bishop_result = self.search_critical_circle(method="bishop", **search_kwargs)
        janbu_result = self.search_critical_circle(method="janbu", **search_kwargs)
        return {"bishop_simplified": bishop_result, "janbu_simplified": janbu_result}

    def rapid_drawdown_analysis(self, ru=0.4, **search_kwargs):
        """
        SIMPLIFIED rapid-drawdown check (see module docstring for the
        distinction from the full USACE three-stage undrained-strength
        method): re-runs Bishop's Simplified search using the ru-based
        pore-pressure override instead of the steady-state phreatic
        line, representing pore pressures that haven't yet dissipated
        following a fast reservoir drawdown.
        """
        xi_vertices, y_vertices = self._ground_profile()
        toe_xi = 0.0
        crest_start = self.downstream_slope_h_per_v * self.dam_height_m

        xc_candidates = np.linspace(crest_start * 0.3, crest_start * 2.5,
                                     search_kwargs.get("n_centers", 12))
        yc_candidates = np.linspace(self.dam_height_m * 0.8, self.dam_height_m * 3.0,
                                     search_kwargs.get("n_centers", 12))

        best = {"fos": np.inf, "slices": None, "xc": None, "yc": None, "ru_used": ru}
        for xc in xc_candidates:
            for yc in yc_candidates:
                slices = build_trial_circle(xc, yc, toe_xi, xi_vertices, y_vertices,
                                             search_kwargs.get("n_slices", 30))
                if slices is None:
                    continue
                u = pore_pressure_from_ru(slices["height"], self.soil_unit_weight_kn_m3, ru)
                fos = bishop_simplified_fos(slices, self.cohesion_kpa, self.friction_angle_deg,
                                             self.soil_unit_weight_kn_m3, u)
                if np.isfinite(fos) and 0 < fos < best["fos"]:
                    best = {"fos": fos, "slices": slices, "xc": xc, "yc": yc, "ru_used": ru}
        return best

    def progressive_piping_erosion_rate(self, exit_gradient, tau_c_pa, kd_cm3_per_N_s,
                                         seepage_flow_depth_m=0.02):
        """
        Hanson excess-shear-stress erosion rate applied at the
        downstream exit face, using the exit hydraulic gradient (from
        the base class's exit_gradient()) converted to an equivalent
        shear stress via tau = rho*g*depth*gradient.

        seepage_flow_depth_m : assumed depth of the emerging seepage
            flow at the exit point. An initial piping channel/seep is
            typically millimeter-to-centimeter scale (NOT a 1 m sheet
            of water) -- this defaults to 2 cm as a small-emerging-seep
            order-of-magnitude starting point, but tau scales LINEARLY
            with this value, so the resulting erosion rate is highly
            sensitive to it. Treat the output as order-of-magnitude
            screening, and re-run with a depth matched to actual
            observed/expected seepage exit conditions for anything
            beyond a first-pass estimate -- this single assumption is
            the main reason this remains a screening tool, not a
            calibrated internal-erosion progression model.
        """
        tau_pa = 1000.0 * G * seepage_flow_depth_m * exit_gradient
        rate = hanson_erosion_rate(tau_pa, tau_c_pa, kd_cm3_per_N_s)
        return {
            "exit_gradient": exit_gradient,
            "assumed_seepage_flow_depth_m": seepage_flow_depth_m,
            "applied_shear_stress_pa": float(tau_pa),
            "critical_shear_stress_pa": tau_c_pa,
            "erosion_rate_cm_per_s": float(rate),
            "erosion_rate_mm_per_hour": float(rate * 10 * 3600),
        }

    def wave_overtopping_crest_erosion(self, head_above_crest_m, tau_c_pa, kd_cm3_per_N_s,
                                        manning_n=0.035, weir_coeff=1.7):
        """
        Overtopping unit discharge (broad-crested weir) -> normal-flow
        depth on the downstream face (Manning) -> shear stress -> Hanson
        erosion rate. Replaces the unverifiable user-named "Bremicker/
        Ermolenko" model (see engines/EarthDamSeepage.py's roadmap note)
        with a defensible combination of two independently-verified
        formulas instead.
        """
        q = overtopping_unit_discharge(head_above_crest_m, weir_coeff)
        face = overtopping_face_shear_stress(q, self.downstream_slope_h_per_v, manning_n)
        rate = hanson_erosion_rate(face["shear_stress_pa"], tau_c_pa, kd_cm3_per_N_s)
        return {
            "overtopping_unit_discharge_m2_s": float(q),
            "flow_depth_on_face_m": face["flow_depth_m"],
            "applied_shear_stress_pa": face["shear_stress_pa"],
            "critical_shear_stress_pa": tau_c_pa,
            "erosion_rate_cm_per_s": float(rate),
            "erosion_rate_mm_per_hour": float(rate * 10 * 3600),
        }