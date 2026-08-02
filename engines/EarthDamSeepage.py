"""
engines/EarthDamSeepage.py

AdvancedEmbankmentEngine -- the 13-feature Geotechnical Workstation,
extending engines.embankment.EarthDamSeepage.
=======================================================================
TECHNICAL BACKGROUND (read before using):
This module intentionally does NOT implement all 13 requested features
with equal depth. Four are implemented here with real, web-verified
formulas: (1) a standard soil-type/permeability/void-ratio lookup table,
(2) anisotropic permeability (kx != kz) via Casagrande's transformed-
section method, which rescales the seepage domain so the existing
isotropic base-parabola solution in engines/embankment.py applies
directly, (3) Terzaghi & Peck's (1948) filter-design criteria
(D15f/D85b <= 4-5 for piping/retention, D15f/D15b >= 4-5 for
permeability, both confirmed via web search against multiple
geotechnical references during writing), and (4) the Seed-Idriss (1971,
refined NCEER 1997) simplified liquefaction procedure comparing
earthquake-induced cyclic stress ratio (CSR) to the SPT-based cyclic
resistance ratio (CRR), with every coefficient below confirmed against
an independently-published worked numerical example during the web
search for this module. The remaining nine features (Bishop's/Janbu's
slope stability, rapid drawdown, progressive piping rate, frost heave,
geosynthetic reinforcement, wave-overtopping erosion, and
saturated-unsaturated transient seepage) are each a substantial
sub-discipline in their own right -- they are stubbed below with a
clear docstring of the intended method and a NotImplementedError,
rather than filled in with unverified formulas, since fabricated
numbers in a dam-safety tool are worse than an honest "not yet built".
"""

from dataclasses import dataclass

import numpy as np

from engines.embankment import (
    EarthDamSeepage, casagrande_focal_parameter, casagrande_base_parabola,
    phreatic_entry_point,
)

G = 9.81


# ===========================================================================
# 1. Soil Type & Permeability Auto-Selector
# ===========================================================================

@dataclass
class SoilTypeProperties:
    name: str
    permeability_mps_range: tuple      # (min, max) hydraulic conductivity, m/s
    void_ratio_range: tuple            # (min, max) typical void ratio
    specific_gravity: float
    d50_mm_range: tuple                # (min, max) typical median grain size, mm


# THEORY: these are the standard order-of-magnitude permeability bands and
# void-ratio ranges reproduced in essentially every soil mechanics
# textbook (e.g. Das, "Principles of Geotechnical Engineering", Table on
# typical coefficient-of-permeability ranges by soil type) -- deliberately
# given as ranges, not single numbers, since actual k varies over 2-3
# orders of magnitude even within one soil classification depending on
# gradation, density, and fabric; a real project must still lab-test the
# actual borrow material.
SOIL_LIBRARY = {
    "gravel":      SoilTypeProperties("Gravel",      (1e-2, 1.0),   (0.30, 0.50), 2.65, (2.0, 75.0)),
    "coarse_sand": SoilTypeProperties("Coarse Sand",  (1e-3, 1e-1),  (0.35, 0.55), 2.65, (0.6, 2.0)),
    "fine_sand":   SoilTypeProperties("Fine Sand",    (1e-5, 1e-3),  (0.40, 0.60), 2.65, (0.075, 0.425)),
    "silty_clay":  SoilTypeProperties("Silty Clay",   (1e-9, 1e-6),  (0.55, 0.90), 2.70, (0.002, 0.075)),
}


def select_soil_properties(soil_key: str) -> SoilTypeProperties:
    """
    Auto-selector lookup: returns typical (k, e, Gs, d50) ranges for a
    named soil type, so a user doesn't need to look up a table by hand
    for a first-pass estimate -- see SOIL_LIBRARY's docstring comment
    above for the caveat that these are ranges, not lab-tested values.
    """
    key = soil_key.strip().lower().replace(" ", "_")
    if key not in SOIL_LIBRARY:
        raise KeyError(f"Unknown soil type '{soil_key}'. Choose from: {list(SOIL_LIBRARY)}")
    return SOIL_LIBRARY[key]


# ===========================================================================
# 2. Anisotropic Permeability (kx != kz) -- Transformed Section Method
# ===========================================================================

def anisotropic_transform_factor(kx_mps, kz_mps):
    """
    THEORY (Casagrande's transformed-section method, standard in every
    seepage/flow-net textbook -- e.g. Cedergren, "Seepage, Drainage, and
    Flow Nets"): real embankments are built in compacted horizontal
    lifts, which almost always makes horizontal permeability kx exceed
    vertical permeability kz (kx/kz commonly 2-10x for compacted clay
    cores). Laplace's governing seepage equation,
        kx * d2h/dx2 + kz * d2h/dz2 = 0,
    is NOT the isotropic Laplace equation unless kx=kz. However,
    substituting a TRANSFORMED horizontal coordinate
        x' = x * sqrt(kz / kx)
    converts it exactly into the isotropic form
        d2h/dx'2 + d2h/dz2 = 0,
    i.e. the SAME base-parabola solution already implemented in
    engines/embankment.py applies directly on the transformed (x', z)
    section -- only the horizontal geometry is rescaled; z (elevation)
    is untouched. The equivalent isotropic permeability for computing
    real (not transformed-section) discharge is
        k_equivalent = sqrt(kx * kz).

    Returns
    -------
    dict with the horizontal scale factor (x' = x * scale), and
    k_equivalent for discharge calculations.
    """
    scale = np.sqrt(kz_mps / kx_mps)
    k_equivalent = np.sqrt(kx_mps * kz_mps)
    return {"horizontal_scale_factor": float(scale), "k_equivalent_mps": float(k_equivalent)}


def anisotropic_phreatic_line(dam_height_m, upstream_water_depth_m,
                               upstream_slope_h_per_v, downstream_slope_h_per_v,
                               crest_width_m, kx_mps, kz_mps, n_points=200):
    """
    Phreatic line for an anisotropic embankment: transforms the
    horizontal geometry by sqrt(kz/kx) (see anisotropic_transform_factor
    above), solves the ALREADY-VERIFIED isotropic base parabola
    (engines.embankment) on that transformed section, then transforms
    the result back to real (untransformed) horizontal coordinates for
    plotting against the true dam cross-section.
    """
    xform = anisotropic_transform_factor(kx_mps, kz_mps)
    scale = xform["horizontal_scale_factor"]

    # Transform ALL horizontal geometry inputs by `scale` before calling
    # the isotropic solver -- slopes (expressed as horizontal-per-
    # vertical ratios) and the crest width are all horizontal
    # quantities, so each is scaled the same way; dam height (vertical)
    # is untouched.
    d, h, L = phreatic_entry_point(
        dam_height_m, upstream_water_depth_m,
        upstream_slope_h_per_v * scale, downstream_slope_h_per_v * scale,
        crest_width_m * scale,
    )
    s0 = casagrande_focal_parameter(d, h)

    xi_transformed = np.linspace(0.0, d, n_points)
    y = casagrande_base_parabola(s0, xi_transformed)

    # Transform back to real horizontal coordinates (divide by `scale`,
    # the inverse of the forward x' = x*scale transform) for plotting
    # against the actual (untransformed) dam cross-section.
    xi_real = xi_transformed / scale

    return {
        "xi_m": xi_real,
        "elevation_m": y,
        "focal_parameter_s0_m": s0,
        "k_equivalent_mps": xform["k_equivalent_mps"],
        "horizontal_scale_factor": scale,
    }


# ===========================================================================
# 3. Filter Criteria & Granular Stability Check (Terzaghi's Filter Rules)
# ===========================================================================

def terzaghi_filter_check(d15_filter_mm, d85_base_mm, d15_base_mm,
                           d50_filter_mm=None, d50_base_mm=None) -> dict:
    """
    Terzaghi & Peck (1948) filter-design criteria, confirmed via web
    search against multiple independent geotechnical references during
    the writing of this module (the classic values are reported as
    ratios of 4 or 5 depending on the source; this implementation uses
    the commonly-adopted <=5 / >=5 bounds and separately reports the
    stricter <=4 / >=4 bounds so a caller can see how close to the
    tighter historical criterion the material sits):

        Piping/retention criterion (filter must not let base material
        wash through its own pore spaces):
            D15(filter) / D85(base) <= 4 to 5

        Permeability criterion (filter must be free-draining relative
        to the base material, so it doesn't itself become a barrier):
            D15(filter) / D15(base) >= 4 to 5

        Gradation/uniformity criterion (avoids segregation and ensures
        the filter isn't drastically coarser overall than the base):
            D50(filter) / D50(base) <= 25   (only checked if D50 given)
    """
    piping_ratio = d15_filter_mm / d85_base_mm
    permeability_ratio = d15_filter_mm / d15_base_mm

    result = {
        "piping_ratio_D15f_D85b": float(piping_ratio),
        "piping_criterion_pass_le5": bool(piping_ratio <= 5.0),
        "piping_criterion_pass_le4_strict": bool(piping_ratio <= 4.0),
        "permeability_ratio_D15f_D15b": float(permeability_ratio),
        "permeability_criterion_pass_ge5": bool(permeability_ratio >= 5.0),
        "permeability_criterion_pass_ge4_strict": bool(permeability_ratio >= 4.0),
    }

    if d50_filter_mm is not None and d50_base_mm is not None:
        gradation_ratio = d50_filter_mm / d50_base_mm
        result["gradation_ratio_D50f_D50b"] = float(gradation_ratio)
        result["gradation_criterion_pass_le25"] = bool(gradation_ratio <= 25.0)

    result["overall_pass"] = bool(
        result["piping_criterion_pass_le5"] and result["permeability_criterion_pass_ge5"]
        and result.get("gradation_criterion_pass_le25", True)
    )
    return result


# ===========================================================================
# 4. Liquefaction & Dynamic Cyclic Mobility Potential (Seed-Idriss)
# ===========================================================================

def stress_reduction_factor_rd(depth_m):
    """
    Liao & Whitman (1986) depth-dependent stress reduction factor rd,
    accounting for the fact that soil is not a perfectly rigid column
    during earthquake shaking (the simplifying assumption Seed & Idriss,
    1971 started from) -- confirmed via web search against multiple
    independent liquefaction-calculator references during the writing
    of this module, each showing the identical piecewise-linear form:

        rd = 1.0 - 0.00765*z          for z <= 9.15 m
        rd = 1.174 - 0.0267*z         for 9.15 m < z <= 23 m
    """
    z = depth_m
    if z <= 9.15:
        return 1.0 - 0.00765 * z
    if z <= 23.0:
        return 1.174 - 0.0267 * z
    # Beyond 23 m the simplified procedure's original calibration range
    # is exceeded; Cetin et al. (2004) extends it further, but this
    # module stays within the original Seed-Idriss/Liao-Whitman range
    # and flags depths beyond it rather than extrapolating silently.
    raise ValueError("Depth exceeds the 23 m calibration range of the Liao & Whitman (1986) rd formula; "
                      "use a depth-specific site response analysis instead of the simplified procedure.")


def cyclic_stress_ratio(amax_g, sigma_v_kpa, sigma_v_eff_kpa, depth_m):
    """
    Seed & Idriss (1971/1982) simplified cyclic stress ratio:

        CSR = 0.65 * (amax/g) * (sigma_v / sigma_v') * rd

    amax_g : peak ground acceleration, as a fraction of g (e.g. 0.25 for
             0.25g)
    sigma_v_kpa, sigma_v_eff_kpa : total and effective vertical
             overburden stress at the depth of interest (kPa)
    depth_m : depth below ground surface (m), for the rd stress-
             reduction factor above

    The 0.65 factor converts the PEAK cyclic shear stress ratio to an
    equivalent UNIFORM cyclic stress ratio representative of the most
    damaging ~65% of cycles over the full irregular earthquake record --
    the original Seed & Idriss (1967) calibration choice, confirmed via
    web search as still universally used in every modern variant of this
    procedure (NCEER 1997, Idriss & Boulanger 2004/2008).
    """
    rd = stress_reduction_factor_rd(depth_m)
    return 0.65 * amax_g * (sigma_v_kpa / sigma_v_eff_kpa) * rd


def cyclic_resistance_ratio_spt(n1_60):
    """
    NCEER (1997) SPT clean-sand base curve for the cyclic resistance
    ratio at Mw=7.5, as a function of the overburden- and energy-
    corrected SPT blow count (N1)60:

        CRR_7.5 = 1/(34-(N1)60) + (N1)60/135 + 50/(10*(N1)60+45)^2 - 1/200

    Confirmed via web search directly against a published worked
    numerical example during the writing of this module: for
    (N1)60 = 15, the formula gives CRR_7.5 = 0.160, matching the
    independent reference exactly.

    Valid for (N1)60 < 30; soils at or above (N1)60 = 30 are considered
    too dense to liquefy under this procedure ("non-liquefiable").
    """
    if n1_60 >= 30:
        return None  # non-liquefiable under the simplified procedure
    return (1.0 / (34 - n1_60) + n1_60 / 135.0
            + 50.0 / (10 * n1_60 + 45) ** 2 - 1.0 / 200.0)


def magnitude_scaling_factor(moment_magnitude):
    """
    Idriss (1999) magnitude scaling factor, converting the Mw=7.5-
    calibrated CRR above to a different earthquake magnitude:

        MSF = 10^2.24 / Mw^2.56

    Returns exactly 1.0 at Mw=7.5 by construction (10^2.24/7.5^2.56 ~
    1.0), consistent with CRR_7.5 already being the Mw=7.5 baseline.
    """
    return 10 ** 2.24 / moment_magnitude ** 2.56


def liquefaction_factor_of_safety(amax_g, depth_m, n1_60, moment_magnitude=7.5,
                                   sigma_v_kpa=None, sigma_v_eff_kpa=None,
                                   unit_weight_kn_m3=18.0, water_table_depth_m=0.0):
    """
    Combined Seed-Idriss simplified liquefaction check: FS = (CRR_7.5 *
    MSF) / CSR. FS < 1.0 indicates liquefaction is predicted to trigger;
    conventional practice (per the worked reference confirmed during
    this module's web search) treats FS < 1.0 as "liquefaction likely",
    with FS in roughly 1.0-1.3 often still flagged as marginal given the
    procedure's inherent uncertainty.

    If sigma_v/sigma_v_eff aren't supplied directly, they are estimated
    from a uniform unit weight and water-table depth (total stress =
    unit_weight * depth; effective stress subtracts hydrostatic pore
    pressure below the water table).
    """
    if sigma_v_kpa is None:
        sigma_v_kpa = unit_weight_kn_m3 * depth_m
    if sigma_v_eff_kpa is None:
        pore_pressure_kpa = 9.81 * max(depth_m - water_table_depth_m, 0.0)
        sigma_v_eff_kpa = sigma_v_kpa - pore_pressure_kpa

    csr = cyclic_stress_ratio(amax_g, sigma_v_kpa, sigma_v_eff_kpa, depth_m)
    crr75 = cyclic_resistance_ratio_spt(n1_60)
    if crr75 is None:
        return {"csr": float(csr), "crr_7_5": None, "factor_of_safety": None,
                "status": "non_liquefiable (N1_60 >= 30)"}

    msf = magnitude_scaling_factor(moment_magnitude)
    crr = crr75 * msf
    fos = crr / csr

    return {
        "csr": float(csr),
        "crr_7_5": float(crr75),
        "magnitude_scaling_factor": float(msf),
        "crr_adjusted": float(crr),
        "factor_of_safety": float(fos),
        "status": ("liquefaction_likely" if fos < 1.0
                    else "marginal" if fos < 1.3 else "not_liquefiable"),
    }


# ===========================================================================
# AdvancedEmbankmentEngine -- ties the verified features above to a dam,
# and documents (without faking) the remaining roadmap items.
# ===========================================================================

class AdvancedEmbankmentEngine(EarthDamSeepage):
    """
    Extends engines.embankment.EarthDamSeepage with the geotechnical
    workstation's advanced analyses. Methods implemented with real,
    web-verified formulas: anisotropic seepage, filter criteria,
    liquefaction. Methods NOT yet implemented raise NotImplementedError
    with a docstring describing the intended method, rather than
    returning fabricated numbers -- see the module docstring for why.
    """

    def __init__(self, *args, kx_mps=None, kz_mps=None, **kwargs):
        super().__init__(*args, **kwargs)
        # If kx/kz aren't given, assume isotropic (kx=kz=hydraulic_conductivity_mps
        # from the base class) so anisotropic methods still work with a
        # 1:1 ratio rather than requiring every caller to specify both.
        self.kx_mps = kx_mps if kx_mps is not None else self.hydraulic_conductivity_mps
        self.kz_mps = kz_mps if kz_mps is not None else self.hydraulic_conductivity_mps

    # -- Implemented, verified features -----------------------------------

    def anisotropic_phreatic_line(self, n_points=200):
        return anisotropic_phreatic_line(
            self.dam_height_m, self.upstream_water_depth_m,
            self.upstream_slope_h_per_v, self.downstream_slope_h_per_v,
            self.crest_width_m, self.kx_mps, self.kz_mps, n_points,
        )

    def check_filter_criteria(self, d15_filter_mm, d85_base_mm, d15_base_mm,
                               d50_filter_mm=None, d50_base_mm=None):
        return terzaghi_filter_check(d15_filter_mm, d85_base_mm, d15_base_mm,
                                      d50_filter_mm, d50_base_mm)

    def liquefaction_check(self, amax_g, depth_m, n1_60, moment_magnitude=7.5, **kwargs):
        return liquefaction_factor_of_safety(amax_g, depth_m, n1_60, moment_magnitude, **kwargs)

    # -- Roadmap: documented, not faked ------------------------------------

    def slope_stability_bishop_janbu(self, *args, **kwargs):
        """
        ROADMAP (not yet implemented). Intended method: Bishop's (1955)
        Simplified method and Janbu's (1954, 1973) Generalized Procedure
        of Slices, run in parallel over a searched grid of trial circular
        (Bishop) / non-circular (Janbu) slip surfaces, each requiring
        slice-by-slice normal/shear force equilibrium solved iteratively
        (Bishop's FoS appears on both sides of its own equation). This
        needs a proper slip-surface search algorithm and slice geometry
        engine -- a substantial standalone module, not a single formula,
        so it is not implemented here yet rather than approximated.
        """
        raise NotImplementedError(self.slope_stability_bishop_janbu.__doc__)

    def rapid_drawdown_analysis(self, *args, **kwargs):
        """
        ROADMAP (not yet implemented). Intended method: Bishop's (1954)
        or Morgenstern's (1963) simplified rapid-drawdown pore-pressure
        procedures, which require the same slip-surface machinery as
        slope_stability_bishop_janbu() above plus a drawdown-rate-
        dependent pore-pressure ratio (ru) applied to each slice -- built
        on top of that method once it exists, not before.
        """
        raise NotImplementedError(self.rapid_drawdown_analysis.__doc__)

    def progressive_piping_erosion_rate(self, *args, **kwargs):
        """
        ROADMAP (not yet implemented). Intended method: Hanson's
        erodibility-index approach or the SIMBA/WinDAM internal-erosion
        progression models, relating local exit hydraulic gradient
        (engines.embankment.exit_gradient_dupuit) and a soil erodibility
        coefficient to a pipe-widening rate over time -- these
        coefficients are soil-specific and typically require a jet-
        erosion test (JET) result as input, which this module has no
        placeholder data source for.
        """
        raise NotImplementedError(self.progressive_piping_erosion_rate.__doc__)

    def frost_heave_pore_pressure(self, *args, **kwargs):
        """
        ROADMAP (not yet implemented). Intended method: a segregation-
        potential (SP) based frost-heave model (Konrad & Morgenstern,
        1980) relating temperature gradient, SP (a soil-specific lab-
        measured parameter), and overburden pressure to heave rate and
        cyclic ice-lens pore-pressure -- not attempted here since SP has
        no reasonable default value across soil types (unlike the
        Shields/Terzaghi parameters used elsewhere in this codebase).
        """
        raise NotImplementedError(self.frost_heave_pore_pressure.__doc__)

    def geosynthetic_reinforcement_pullout(self, *args, **kwargs):
        """
        ROADMAP (not yet implemented). Intended method: FHWA geosynthetic
        pullout resistance, Pr = 2*Le*sigma_v*tan(phi)*Ci*alpha (interaction
        coefficient Ci and scale factor alpha are product-specific,
        manufacturer-published values) -- not stubbed with a fabricated
        Ci/alpha since those genuinely vary by geosynthetic product line.
        """
        raise NotImplementedError(self.geosynthetic_reinforcement_pullout.__doc__)

    def wave_overtopping_crest_erosion(self, *args, **kwargs):
        """
        ROADMAP (not yet implemented). The user-specified "Bremicker/
        Ermolenko model" could not be independently verified during the
        web search for this module -- rather than implement an unverified
        named formula, this is left unimplemented. A defensible
        alternative once revisited: overtopping unit discharge via the
        broad-crested weir relation already implemented in
        engines.dam_break.breach_outflow_hydrograph(), combined with
        Hanson's erodibility-index method (see
        progressive_piping_erosion_rate above) for the erosion-rate half
        of the problem.
        """
        raise NotImplementedError(self.wave_overtopping_crest_erosion.__doc__)

    def saturated_unsaturated_transient_seepage(self, *args, **kwargs):
        """
        ROADMAP (not yet implemented). Intended method: Richards'
        equation for variably-saturated flow above the phreatic surface,
        needing a soil-water characteristic curve (van Genuchten or
        Brooks-Corey parameters) and typically solved via finite-element/
        finite-difference time-stepping (e.g. as in SEEP/W) -- a
        fundamentally different (PDE time-stepping) solver architecture
        from the closed-form Casagrande approach used elsewhere in this
        module, and out of scope for a single-function addition.
        """
        raise NotImplementedError(self.saturated_unsaturated_transient_seepage.__doc__)