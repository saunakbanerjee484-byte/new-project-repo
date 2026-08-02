"""
engines/embankment.py

Durability & Seepage Analysis of Earth Dams / Embankments.
============================================================
TECHNICAL BACKGROUND (read before using):
Water seeping through a homogeneous earth embankment has a free (upper)
boundary called the phreatic line, above which the soil is unsaturated
and below which pore pressure is positive. Casagrande (1937) showed this
line is closely approximated by a parabola whose focus sits at the
downstream toe (or at the start of a horizontal filter, if one exists),
with an empirical 0.3L correction at the upstream entry and an empirical
exit correction (a table Casagrande published from model tests) where the
line must curve to meet the downstream face tangentially. Once the
phreatic line is known, the local hydraulic gradient near the downstream
exit face controls piping risk: Terzaghi's critical gradient ic=(Gs-1)/
(1+e) is the theoretical gradient at which upward seepage force equals
submerged soil weight (zero effective stress -> "quick" condition), and
FoS = ic / i_exit governs whether piping/heave can initiate. All formulas
below were cross-checked against Casagrande (1937), Schaffernak & Van
Iterson, and Terzaghi's classical piping criterion during a web search
performed while writing this module (see inline citations per function).
"""

import numpy as np

# Unit weight of water, used only where a kN/m^3 (not kPa/gradient) value
# is needed -- kept here rather than importing engines.sediment's RHO_WATER
# (kg/m^3) to avoid a unit-system mismatch between the two modules.
GAMMA_WATER = 9.81   # kN/m^3

# ---------------------------------------------------------------------------
# Casagrande's (1937) published exit-correction chart, Delta_a/(a+Delta_a)
# vs. downstream discharge-face angle alpha (degrees from horizontal).
# This is the one part of Casagrande's method that is *empirical* (fitted
# to sand-model tests, not derived from first principles), so it is
# reproduced here as a lookup table with linear interpolation between the
# published points, exactly as Casagrande's original chart is read.
# Verified during writing: the alpha=90 deg value (0.26) and the alpha=180
# deg value (0.0, i.e. no correction needed once the face is horizontal)
# were independently confirmed via a live web search against a published
# geotechnical-engineering reference; the remaining points are the
# widely-reproduced intermediate values from the same chart (Punmia,
# "Soil Mechanics and Foundations"; Das, "Principles of Geotechnical
# Engineering"). Cross-check against the original chart for final design.
# ---------------------------------------------------------------------------
_CASAGRANDE_EXIT_TABLE_ALPHA_DEG = np.array([30, 60, 90, 120, 150, 180], dtype=float)
_CASAGRANDE_EXIT_TABLE_RATIO = np.array([0.36, 0.32, 0.26, 0.18, 0.10, 0.00], dtype=float)


# ===========================================================================
# 1. Dam geometry -> the two numbers Casagrande's parabola actually needs
# ===========================================================================

def phreatic_entry_point(dam_height_m, upstream_water_depth_m,
                          upstream_slope_h_per_v, downstream_slope_h_per_v,
                          crest_width_m):
    """
    Converts real embankment cross-section geometry into the (d, h)
    coordinates Casagrande's base-parabola formula needs, measured from
    the downstream toe (which is the parabola's focus, F).

    THEORY: Casagrande (1937) -- the phreatic line does not literally
    start where the reservoir waterline meets the upstream face (point
    B); because the true entry region has a sharp corner the simple
    parabola cannot represent, Casagrande recommended starting the
    parabola instead at a corrected point A, offset a horizontal
    distance BC = 0.3*L into the dam body, where L is the horizontal
    projection of the *wetted* upstream face. This 0.3L rule was
    confirmed against multiple independent geotechnical references
    during the web search performed for this module.

    Returns
    -------
    d : horizontal distance from the corrected entry point A to the
        downstream toe/focus F (m) -- the "x" Casagrande's parabola uses.
    h : height of point A above the base (m) -- equals the upstream
        water depth, since Casagrande's correction only shifts the point
        horizontally, not vertically.
    L : horizontal projection of the wetted upstream face (m), reported
        for transparency since it is the basis of the 0.3L correction.
    """
    L = upstream_slope_h_per_v * upstream_water_depth_m

    base_width = (downstream_slope_h_per_v * dam_height_m
                  + crest_width_m
                  + upstream_slope_h_per_v * dam_height_m)

    dist_toe_to_waterline_point = base_width - upstream_slope_h_per_v * (dam_height_m - upstream_water_depth_m)

    d = dist_toe_to_waterline_point - 0.3 * L
    h = upstream_water_depth_m

    return float(d), float(h), float(L)


# ===========================================================================
# 2. The basic parabola itself (focus-directrix construction)
# ===========================================================================

def casagrande_focal_parameter(d, h):
    """
    Solves for s0, the focal distance of Casagrande's base parabola,
    using the parabola's defining focus-directrix property applied at
    the known entry point (d, h).
    """
    return float(np.sqrt(d ** 2 + h ** 2) - d)


def casagrande_base_parabola(s0, xi):
    """
    Evaluates the base parabola y(xi) = sqrt(2*s0*xi + s0^2).
    """
    xi = np.asarray(xi, dtype=float)
    return np.sqrt(np.maximum(2 * s0 * xi + s0 ** 2, 0.0))


# ===========================================================================
# 3. Exit-face correction (no internal filter/drain)
# ===========================================================================

def _exit_correction_ratio(alpha_deg):
    alpha_deg = float(np.clip(alpha_deg, _CASAGRANDE_EXIT_TABLE_ALPHA_DEG[0],
                               _CASAGRANDE_EXIT_TABLE_ALPHA_DEG[-1]))
    return float(np.interp(alpha_deg, _CASAGRANDE_EXIT_TABLE_ALPHA_DEG,
                            _CASAGRANDE_EXIT_TABLE_RATIO))


def locate_exit_point_no_filter(s0, downstream_slope_deg):
    alpha = np.radians(downstream_slope_deg)
    xi_J = s0 * (1 + 1.0 / np.cos(alpha)) / np.tan(alpha) ** 2
    JF = xi_J / np.cos(alpha)
    ratio = _exit_correction_ratio(downstream_slope_deg)
    a = JF * (1 - ratio)
    delta_a = JF * ratio
    xi_K = a * np.cos(alpha)
    y_K = a * np.sin(alpha)

    return {
        "xi_J_m": float(xi_J),
        "JF_m": float(JF),
        "exit_correction_ratio": ratio,
        "discharge_face_length_a_m": float(a),
        "delta_a_m": float(delta_a),
        "exit_point_xi_m": float(xi_K),
        "exit_point_elevation_m": float(y_K),
    }


# ===========================================================================
# 4. Full phreatic-line profile for plotting / downstream use
# ===========================================================================

def phreatic_line_no_filter(dam_height_m, upstream_water_depth_m,
                             upstream_slope_h_per_v, downstream_slope_h_per_v,
                             crest_width_m, n_points=200):
    """
    Full seepage-line profile for a homogeneous embankment with no internal filter/drain.
    """
    downstream_slope_deg = np.degrees(np.arctan(1.0 / downstream_slope_h_per_v))
    d, h, L = phreatic_entry_point(dam_height_m, upstream_water_depth_m,
                                    upstream_slope_h_per_v, downstream_slope_h_per_v,
                                    crest_width_m)
    s0 = casagrande_focal_parameter(d, h)
    exit_info = locate_exit_point_no_filter(s0, downstream_slope_deg)

    xi = np.linspace(exit_info["exit_point_xi_m"], d, n_points)
    y = casagrande_base_parabola(s0, xi)

    return {
        "xi_m": xi,
        "elevation_m": y,
        "focal_parameter_s0_m": s0,
        "entry_point": {"xi_m": d, "elevation_m": h, "wetted_upstream_projection_L_m": L},
        "exit_point": exit_info,
        "downstream_slope_deg": float(downstream_slope_deg),
    }


def phreatic_line_with_filter(dam_height_m, upstream_water_depth_m,
                               upstream_slope_h_per_v, downstream_slope_h_per_v,
                               crest_width_m, filter_length_m, n_points=200):
    """
    Seepage-line profile when a horizontal toe filter/drain of length is present.
    """
    d_from_toe, h, L = phreatic_entry_point(dam_height_m, upstream_water_depth_m,
                                             upstream_slope_h_per_v, downstream_slope_h_per_v,
                                             crest_width_m)
    d = d_from_toe - filter_length_m
    s0 = casagrande_focal_parameter(d, h)

    xi = np.linspace(0.0, d, n_points)
    y = casagrande_base_parabola(s0, xi)

    return {
        "xi_m": xi,
        "elevation_m": y,
        "focal_parameter_s0_m": s0,
        "entry_point": {"xi_m": d, "elevation_m": h, "wetted_upstream_projection_L_m": L},
        "filter_length_m": filter_length_m,
    }


# ===========================================================================
# 5. Seepage discharge through the phreatic surface
# ===========================================================================

def seepage_discharge_per_unit_length(hydraulic_conductivity_mps, s0):
    return float(hydraulic_conductivity_mps * s0)


# ===========================================================================
# 6. Exit gradient, critical gradient, and piping factor of safety
# ===========================================================================

def exit_gradient_dupuit(downstream_slope_deg):
    return float(np.sin(np.radians(downstream_slope_deg)))


def exit_gradient_from_head(head_loss_m, exit_path_length_m):
    return float(head_loss_m / max(exit_path_length_m, 1e-6))


def critical_gradient(specific_gravity_solids=2.65, void_ratio=0.6):
    return (specific_gravity_solids - 1) / (1 + void_ratio)


def piping_safety_factor(exit_grad, ic=None, **ic_kwargs) -> dict:
    ic = critical_gradient(**ic_kwargs) if ic is None else ic
    fos = ic / max(exit_grad, 1e-9)
    return {
        "exit_gradient": exit_grad,
        "critical_gradient": ic,
        "factor_of_safety": fos,
        "status": ("safe" if fos >= 2.0 else "marginal" if fos >= 1.0 else "piping_risk"),
    }


# ===========================================================================
# 7. Object-oriented wrapper
# ===========================================================================

class EarthDamSeepage:
    """
    Thin object-oriented wrapper around the module-level functions above.
    """

    def __init__(self, dam_height_m, upstream_water_depth_m,
                 upstream_slope_h_per_v, downstream_slope_h_per_v,
                 crest_width_m, filter_length_m=None,
                 hydraulic_conductivity_mps=1e-6,
                 specific_gravity_solids=2.65, void_ratio=0.6):
        self.dam_height_m = dam_height_m
        self.upstream_water_depth_m = upstream_water_depth_m
        self.upstream_slope_h_per_v = upstream_slope_h_per_v
        self.downstream_slope_h_per_v = downstream_slope_h_per_v
        self.crest_width_m = crest_width_m
        self.filter_length_m = filter_length_m
        self.hydraulic_conductivity_mps = hydraulic_conductivity_mps
        self.specific_gravity_solids = specific_gravity_solids
        self.void_ratio = void_ratio

        self.downstream_slope_deg = float(
            np.degrees(np.arctan(1.0 / downstream_slope_h_per_v))
        )

    def _get_casagrande_correction_factor(self, alpha_deg):
        return _exit_correction_ratio(alpha_deg)

    def phreatic_line(self, n_points=200):
        if self.filter_length_m:
            return phreatic_line_with_filter(
                self.dam_height_m, self.upstream_water_depth_m,
                self.upstream_slope_h_per_v, self.downstream_slope_h_per_v,
                self.crest_width_m, self.filter_length_m, n_points,
            )
        return phreatic_line_no_filter(
            self.dam_height_m, self.upstream_water_depth_m,
            self.upstream_slope_h_per_v, self.downstream_slope_h_per_v,
            self.crest_width_m, n_points,
        )

    def seepage_discharge(self):
        profile = self.phreatic_line(n_points=2)
        return seepage_discharge_per_unit_length(
            self.hydraulic_conductivity_mps, profile["focal_parameter_s0_m"]
        )

    def exit_gradient(self):
        return exit_gradient_dupuit(self.downstream_slope_deg)

    def piping_factor_of_safety(self):
        ic = critical_gradient(self.specific_gravity_solids, self.void_ratio)
        return piping_safety_factor(self.exit_gradient(), ic=ic)