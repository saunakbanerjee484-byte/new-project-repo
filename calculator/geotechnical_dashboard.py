"""
calculator/geotechnical_dashboard.py

The Geotechnical Workstation -- interactive Streamlit panel for
engines.embankment / engines.EarthDamSeepage.

Visual identity: the rest of DeltaPulse runs a bluish-white
glassmorphism theme (appropriate for hydrograph/water panels). This tab
is deliberately different -- it layers a warm amber/terracotta earth
palette ON TOP of the same frosted-glass mechanic, so soil/geotechnical
content reads distinctly from pure-hydraulics content while still
feeling like the same app. A thin blue-to-brown gradient accent bar
(water -> earth) marks each section header specifically because this
tab sits at the intersection of geotechnics and hydraulics -- seepage
*through* soil -- rather than being purely one or the other.

Implemented, real, verified panels:
    - Phreatic line visualizer (isotropic + anisotropic overlay)
    - Soil type / permeability auto-selector
    - Terzaghi filter-criteria checker
    - Seed-Idriss liquefaction calculator
    - Piping factor of safety (from the base EarthDamSeepage class)
    - Slope Stability & Erosion (Bishop/Janbu, Drawdown, Piping, Overtopping)
    - Transient Seepage (1D & 2D Richards' Equation Dynamics)

Run standalone: streamlit run calculator/geotechnical_dashboard.py
Or import render_geotechnical_tab() into app.py as a tab.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from engines.embankment import phreatic_line_no_filter, phreatic_line_with_filter
from engines.EarthDamSeepage import (
    AdvancedEmbankmentEngine, SOIL_LIBRARY, select_soil_properties,
    anisotropic_phreatic_line, terzaghi_filter_check,
    liquefaction_factor_of_safety,
)

# Robust Import Fix for VS Code (Pylance) and Streamlit Execution
try:
    # When imported from app.py at the project root
    from calculator.slope_stability_dashboard import render_slope_stability_tab
    from calculator.seepage_transient_dashboard import render_transient_seepage_tab
except ImportError:
    # When run standalone or if VS Code path is acting up
    from slope_stability_dashboard import render_slope_stability_tab
    from seepage_transient_dashboard import render_transient_seepage_tab


# ---------------------------------------------------------------------------
# Theme: warm amber/terracotta glassmorphism, layered on the app's base
# bluish-white glass mechanic. Colors exactly as specified, plus one
# added accent (a water-to-earth gradient) to signal "this tab bridges
# hydraulics and geotechnics".
# ---------------------------------------------------------------------------
AMBER_LIGHT = "#fef3c7"       # warning zones / soil dry state
EARTH_BROWN = "#78350f"       # embankment body / soil layers (deep)
EARTH_TERRACOTTA = "#b45309"  # embankment body / soil layers (mid)
GLASS_BACKDROP = "rgba(254, 243, 199, 0.45)"  # warm frosted glass over the blue base
WATER_ACCENT = "#0891b2"      # added: signals the hydraulics half of this tab

_CSS = f"""
<style>
.geo-section-header {{
    font-weight: 700;
    font-size: 1.15rem;
    padding-bottom: 6px;
    margin-top: 1.2rem;
    border-bottom: 3px solid transparent;
    border-image: linear-gradient(90deg, {WATER_ACCENT}, {EARTH_TERRACOTTA}) 1;
    color: {EARTH_BROWN};
}}
.geo-glass-card {{
    background: {GLASS_BACKDROP};
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(180, 83, 9, 0.25);
    border-radius: 14px;
    padding: 14px 18px;
    margin-bottom: 10px;
}}
.geo-pass {{
    background: rgba(22, 163, 74, 0.15);
    border: 1px solid rgba(22, 163, 74, 0.4);
    border-radius: 10px;
    padding: 8px 12px;
    color: #14532d;
    font-weight: 600;
}}
.geo-fail {{
    background: rgba(220, 38, 38, 0.12);
    border: 1px solid rgba(220, 38, 38, 0.4);
    border-radius: 10px;
    padding: 8px 12px;
    color: #7f1d1d;
    font-weight: 600;
}}
.geo-roadmap-card {{
    background: rgba(120, 53, 15, 0.06);
    border: 1px dashed rgba(120, 53, 15, 0.35);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
    color: {EARTH_BROWN};
    font-size: 0.9rem;
}}
</style>
"""


def _section_header(text):
    st.markdown(f'<div class="geo-section-header">{text}</div>', unsafe_allow_html=True)


def _pass_fail_badge(label, passed):
    cls = "geo-pass" if passed else "geo-fail"
    icon = "✅" if passed else "❌"
    st.markdown(f'<div class="{cls}">{icon} {label}</div>', unsafe_allow_html=True)


def _dam_cross_section_trace(dam_height, m_u, m_d, crest, x_offset=0.0):
    """Simple embankment outline (heel -> crest -> toe) for the plot background, in downstream-toe-referenced xi coordinates."""
    base_width = m_d * dam_height + crest + m_u * dam_height
    xi = np.array([0, m_d * dam_height, m_d * dam_height + crest, base_width]) + x_offset
    z = np.array([0, dam_height, dam_height, 0])
    return xi, z


def render_geotechnical_tab():
    st.markdown(_CSS, unsafe_allow_html=True)
    st.subheader("🟤 Geotechnical Workstation -- Earth Dam / Embankment Seepage")
    st.caption(
        "Soil mechanics meets hydraulics: phreatic line, filter design, liquefaction, "
        "piping analysis, transient seepage, and slope stability for earth embankments -- engines.embankment / engines.EarthDamSeepage."
    )

    with st.sidebar:
        st.header("Dam Geometry")
        dam_height = st.slider("Dam height (m)", 5.0, 60.0, 20.0, key="geo_h")
        water_depth = st.slider("Reservoir water depth (m)", 1.0, dam_height, 18.0, key="geo_wd")
        m_u = st.slider("Upstream slope (H per 1V)", 1.5, 5.0, 3.0, key="geo_mu")
        m_d = st.slider("Downstream slope (H per 1V)", 1.5, 5.0, 2.0, key="geo_md")
        crest = st.slider("Crest width (m)", 3.0, 15.0, 6.0, key="geo_crest")
        has_filter = st.checkbox("Has horizontal toe filter/drain", value=False, key="geo_filter_on")
        filter_len = st.slider("Filter length (m)", 2.0, 40.0, 10.0, key="geo_filter_len") if has_filter else None

    tab_phreatic, tab_soil, tab_filter, tab_liq, tab_piping, tab_slope_stability, tab_transient, tab_roadmap = st.tabs(
        ["Phreatic Line", "Soil Selector", "Filter Criteria", "Liquefaction", "Piping FoS", "Slope Stability & Erosion", "Transient Seepage", "Roadmap"]
    )

    # =======================================================================
    # Tab: Phreatic line (isotropic vs anisotropic overlay)
    # =======================================================================
    with tab_phreatic:
        _section_header("Casagrande Phreatic Line -- Isotropic vs Anisotropic")
        col_a, col_b = st.columns(2)
        with col_a:
            k_mps = st.number_input("Hydraulic conductivity k (m/s)", value=1e-6, format="%.2e", key="geo_k")
        with col_b:
            aniso_on = st.checkbox("Show anisotropic (kx ≠ kz) overlay", value=True, key="geo_aniso_on")
            if aniso_on:
                kx = st.number_input("kx, horizontal (m/s)", value=5e-6, format="%.2e", key="geo_kx")
                kz = st.number_input("kz, vertical (m/s)", value=1e-6, format="%.2e", key="geo_kz")

        if has_filter:
            profile = phreatic_line_with_filter(dam_height, water_depth, m_u, m_d, crest, filter_len)
        else:
            profile = phreatic_line_no_filter(dam_height, water_depth, m_u, m_d, crest)

        base_xi, base_z = _dam_cross_section_trace(dam_height, m_u, m_d, crest)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=base_xi, y=base_z, name="Embankment outline",
                                  line=dict(color=EARTH_BROWN, width=3), fill="tozeroy",
                                  fillcolor="rgba(180,83,9,0.15)"))
        fig.add_trace(go.Scatter(x=profile["xi_m"], y=profile["elevation_m"], name="Phreatic line (isotropic)",
                                  line=dict(color=WATER_ACCENT, width=3)))

        if aniso_on:
            aniso_profile = anisotropic_phreatic_line(dam_height, water_depth, m_u, m_d, crest, kx, kz)
            fig.add_trace(go.Scatter(x=aniso_profile["xi_m"], y=aniso_profile["elevation_m"],
                                      name=f"Phreatic line (anisotropic, kx/kz={kx/kz:.1f})",
                                      line=dict(color="#dc2626", width=2, dash="dash")))
            st.caption(f"Equivalent isotropic permeability sqrt(kx·kz) = {aniso_profile['k_equivalent_mps']:.2e} m/s")

        fig.update_layout(title="Dam Cross-Section & Seepage (Phreatic) Line", xaxis_title="Distance from toe (m)",
                           yaxis_title="Elevation (m)", height=450, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        engine = AdvancedEmbankmentEngine(dam_height, water_depth, m_u, m_d, crest,
                                           filter_length_m=filter_len, hydraulic_conductivity_mps=k_mps)
        q = engine.seepage_discharge()
        st.markdown(f'<div class="geo-glass-card">Seepage discharge per unit length, '
                    f'q = k·s₀ = <b>{q:.3e} m²/s</b> ({q*86400:.4f} m²/day)</div>', unsafe_allow_html=True)

    # =======================================================================
    # Tab: Soil selector
    # =======================================================================
    with tab_soil:
        _section_header("Soil Type & Permeability Auto-Selector")
        soil_choice = st.selectbox("Soil type", list(SOIL_LIBRARY.keys()),
                                    format_func=lambda k: SOIL_LIBRARY[k].name, key="geo_soil_choice")
        props = select_soil_properties(soil_choice)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Permeability k (m/s)", f"{props.permeability_mps_range[0]:.0e} - {props.permeability_mps_range[1]:.0e}")
        c2.metric("Void ratio e", f"{props.void_ratio_range[0]:.2f} - {props.void_ratio_range[1]:.2f}")
        c3.metric("Specific gravity Gs", f"{props.specific_gravity:.2f}")
        c4.metric("D50 (mm)", f"{props.d50_mm_range[0]:.3g} - {props.d50_mm_range[1]:.3g}")
        st.caption("Typical ranges only (textbook order-of-magnitude bands) -- lab-test the actual borrow material for final design.")

    # =======================================================================
    # Tab: Filter criteria
    # =======================================================================
    with tab_filter:
        _section_header("Terzaghi Filter-Criteria Checker")
        c1, c2, c3 = st.columns(3)
        d15f = c1.number_input("D15 of filter (mm)", value=2.0, key="geo_d15f")
        d85b = c2.number_input("D85 of base soil (mm)", value=0.5, key="geo_d85b")
        d15b = c3.number_input("D15 of base soil (mm)", value=0.3, key="geo_d15b")
        include_gradation = st.checkbox("Also check D50 gradation ratio", value=False, key="geo_grad_on")
        d50f = d50b = None
        if include_gradation:
            c4, c5 = st.columns(2)
            d50f = c4.number_input("D50 of filter (mm)", value=3.0, key="geo_d50f")
            d50b = c5.number_input("D50 of base soil (mm)", value=0.2, key="geo_d50b")

        result = terzaghi_filter_check(d15f, d85b, d15b, d50f, d50b)
        _pass_fail_badge(f"Piping/retention: D15f/D85b = {result['piping_ratio_D15f_D85b']:.2f} (need ≤ 5)",
                          result["piping_criterion_pass_le5"])
        _pass_fail_badge(f"Permeability: D15f/D15b = {result['permeability_ratio_D15f_D15b']:.2f} (need ≥ 5)",
                          result["permeability_criterion_pass_ge5"])
        if include_gradation:
            _pass_fail_badge(f"Gradation: D50f/D50b = {result['gradation_ratio_D50f_D50b']:.2f} (need ≤ 25)",
                              result["gradation_criterion_pass_le25"])
        st.markdown("---")
        _pass_fail_badge("Overall filter design", result["overall_pass"])

    # =======================================================================
    # Tab: Liquefaction
    # =======================================================================
    with tab_liq:
        _section_header("Seed-Idriss Simplified Liquefaction Check")
        c1, c2, c3 = st.columns(3)
        amax_g = c1.slider("Peak ground accel. amax (g)", 0.05, 0.6, 0.25, key="geo_amax")
        depth_m = c2.slider("Depth of layer (m)", 1.0, 22.0, 6.0, key="geo_depth")
        n1_60 = c3.slider("Corrected SPT blow count (N1)60", 1, 40, 15, key="geo_n1_60")
        c4, c5 = st.columns(2)
        mw = c4.slider("Earthquake moment magnitude Mw", 5.0, 8.5, 7.5, key="geo_mw")
        wt_depth = c5.slider("Water table depth (m)", 0.0, depth_m, 2.0, key="geo_wt")
        unit_wt = st.slider("Bulk unit weight (kN/m³)", 15.0, 22.0, 18.0, key="geo_unitwt")

        liq = liquefaction_factor_of_safety(amax_g, depth_m, n1_60, mw,
                                             unit_weight_kn_m3=unit_wt, water_table_depth_m=wt_depth)
        if liq["crr_7_5"] is None:
            st.markdown(f'<div class="geo-pass">✅ Non-liquefiable: (N1)60 = {n1_60} ≥ 30 (too dense)</div>',
                        unsafe_allow_html=True)
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("CSR", f"{liq['csr']:.3f}")
            c2.metric("CRR (Mw-adjusted)", f"{liq['crr_adjusted']:.3f}")
            c3.metric("Factor of Safety", f"{liq['factor_of_safety']:.2f}")
            status_map = {"liquefaction_likely": ("❌ Liquefaction likely (FS < 1.0)", "geo-fail"),
                          "marginal": ("⚠️ Marginal (1.0 ≤ FS < 1.3)", "geo-fail"),
                          "not_liquefiable": ("✅ Not liquefiable (FS ≥ 1.3)", "geo-pass")}
            msg, cls = status_map[liq["status"]]
            st.markdown(f'<div class="{cls}">{msg}</div>', unsafe_allow_html=True)

    # =======================================================================
    # Tab: Piping FoS (base EarthDamSeepage class, inherited)
    # =======================================================================
    with tab_piping:
        _section_header("Exit Gradient & Piping Factor of Safety")
        c1, c2 = st.columns(2)
        gs = c1.slider("Specific gravity of solids Gs", 2.5, 2.8, 2.65, key="geo_gs")
        e = c2.slider("Void ratio e", 0.3, 1.0, 0.6, key="geo_e")

        engine2 = AdvancedEmbankmentEngine(dam_height, water_depth, m_u, m_d, crest,
                                            filter_length_m=filter_len,
                                            specific_gravity_solids=gs, void_ratio=e)
        if has_filter:
            st.markdown('<div class="geo-pass">✅ Horizontal filter present -- no exit-face piping check needed '
                        '(seepage exits through the free-draining filter, not the slope face).</div>',
                        unsafe_allow_html=True)
        else:
            fos = engine2.piping_factor_of_safety()
            c1, c2, c3 = st.columns(3)
            c1.metric("Exit gradient", f"{fos['exit_gradient']:.3f}")
            c2.metric("Critical gradient ic", f"{fos['critical_gradient']:.3f}")
            c3.metric("Factor of Safety", f"{fos['factor_of_safety']:.2f}")
            cls = "geo-pass" if fos["status"] == "safe" else "geo-fail"
            st.markdown(f'<div class="{cls}">Status: {fos["status"].upper()}</div>', unsafe_allow_html=True)

    # =======================================================================
    # Tab: Slope Stability & Erosion
    # =======================================================================
    with tab_slope_stability:
        render_slope_stability_tab()

    # =======================================================================
    # Tab: Transient Seepage (1D & 2D Richards' Equation Dynamics)
    # =======================================================================
    with tab_transient:
        render_transient_seepage_tab()

    # =======================================================================
    # Tab: Roadmap (honest, not faked) - Updated
    # =======================================================================
    with tab_roadmap:
        _section_header("Roadmap -- Not Yet Implemented")
        st.caption(
            "These need their own dedicated solvers (PDE time-stepping, "
            "product-specific parameters) rather than a single formula -- listed honestly as "
            "planned instead of filled in with unverified numbers."
        )
        roadmap_items = [
            ("Geosynthetic reinforcement pullout",
             "FHWA pullout formula, needs manufacturer-specific interaction coefficients"),
        ]
        for title, note in roadmap_items:
            st.markdown(f'<div class="geo-roadmap-card"><b>🔜 {title}</b><br>{note}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    st.set_page_config(page_title="Geotechnical Workstation", layout="wide")
    render_geotechnical_tab()