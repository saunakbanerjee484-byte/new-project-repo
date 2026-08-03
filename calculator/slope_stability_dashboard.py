"""
calculator/slope_stability_dashboard.py

Streamlit UI panel for the Slope Stability & Erosion tab inside the
Geotechnical Workstation (calculator/geotechnical_dashboard.py).

This module provides render_slope_stability_tab(), which wraps the
numerics in calculator/slope_stability_engine.py (Bishop's Simplified,
Janbu's Simplified, rapid-drawdown, progressive piping, wave-overtopping
erosion) into interactive Streamlit widgets with Plotly visualizations,
using the same warm amber/terracotta glassmorphism visual identity as the
parent geotechnical_dashboard.

Imported by geotechnical_dashboard.py as:
    from calculator.slope_stability_dashboard import render_slope_stability_tab
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from engines.embankment import phreatic_line_no_filter
from calculator.slope_stability_engine import (
    AdvancedSlopeStabilityEngine,
    embankment_ground_profile,
    build_trial_circle,
    _circle_lower_arc,
    hanson_erosion_rate,
    overtopping_unit_discharge,
    overtopping_face_shear_stress,
)

# ---------------------------------------------------------------------------
# Theme constants -- reused from geotechnical_dashboard.py for visual
# consistency across the Geotechnical Workstation tabs.
# ---------------------------------------------------------------------------
AMBER_LIGHT = "#fef3c7"
EARTH_BROWN = "#78350f"
EARTH_TERRACOTTA = "#b45309"
GLASS_BACKDROP = "rgba(254, 243, 199, 0.45)"
WATER_ACCENT = "#0891b2"

PASS_COLOR = "rgba(22, 163, 74, 0.15)"
FAIL_COLOR = "rgba(220, 38, 38, 0.12)"


def _section_header(text: str):
    """Amber-to-teal gradient-underlined section header."""
    st.markdown(
        f'<div style="font-weight:700;font-size:1.15rem;padding-bottom:6px;'
        f'margin-top:1.2rem;border-bottom:3px solid transparent;'
        f'border-image:linear-gradient(90deg,{WATER_ACCENT},{EARTH_TERRACOTTA}) 1;'
        f'color:{EARTH_BROWN};">{text}</div>',
        unsafe_allow_html=True,
    )


def _glass_card(content_html: str):
    st.markdown(
        f'<div style="background:{GLASS_BACKDROP};backdrop-filter:blur(10px);'
        f'-webkit-backdrop-filter:blur(10px);border:1px solid rgba(180,83,9,0.25);'
        f'border-radius:14px;padding:14px 18px;margin-bottom:10px;">'
        f'{content_html}</div>',
        unsafe_allow_html=True,
    )


def _pass_fail_badge(label: str, passed: bool):
    bg = PASS_COLOR if passed else FAIL_COLOR
    border_c = "rgba(22,163,74,0.4)" if passed else "rgba(220,38,38,0.4)"
    text_c = "#14532d" if passed else "#7f1d1d"
    icon = "✅" if passed else "❌"
    st.markdown(
        f'<div style="background:{bg};border:1px solid {border_c};border-radius:10px;'
        f'padding:8px 12px;color:{text_c};font-weight:600;">{icon} {label}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Plotly helper: embankment cross-section + critical slip circle
# ---------------------------------------------------------------------------

def _plot_cross_section_with_circle(dam_height, m_u, m_d, crest_width,
                                     water_depth, result, method_label,
                                     phreatic):
    """
    Plotly figure showing the embankment outline, phreatic line, and the
    critical slip circle found by the grid search (if any).
    """
    xi_v, y_v = embankment_ground_profile(dam_height, m_u, m_d, crest_width)

    fig = go.Figure()

    # Embankment fill
    fig.add_trace(go.Scatter(
        x=xi_v, y=y_v, name="Embankment",
        line=dict(color=EARTH_BROWN, width=3),
        fill="tozeroy", fillcolor="rgba(180,83,9,0.15)",
    ))

    # Phreatic line
    fig.add_trace(go.Scatter(
        x=phreatic["xi_m"], y=phreatic["elevation_m"],
        name="Phreatic line",
        line=dict(color=WATER_ACCENT, width=2, dash="dash"),
    ))

    # Critical slip circle (if found)
    if result["slices"] is not None:
        xc, yc = result["xc"], result["yc"]
        R = result["slices"]["R"]
        theta = np.linspace(0, 2 * np.pi, 200)
        cx = xc + R * np.cos(theta)
        cy = yc - R * np.sin(theta)
        # Only show the lower arc within the embankment's x-range
        mask = (cy >= -0.5) & (cx >= -2) & (cx <= xi_v[-1] + 5)
        fig.add_trace(go.Scatter(
            x=cx[mask], y=cy[mask],
            name=f"Critical circle (FoS={result['fos']:.2f})",
            line=dict(color="#dc2626", width=2),
            mode="lines",
        ))

        # Slice bases
        fig.add_trace(go.Scatter(
            x=result["slices"]["x_mid"],
            y=result["slices"]["y_circle_mid"],
            name="Slice bases",
            mode="markers",
            marker=dict(size=4, color="#dc2626"),
        ))

    fos_text = f"{result['fos']:.2f}" if np.isfinite(result["fos"]) else "N/A"
    fig.update_layout(
        title=f"{method_label} — Critical FoS = {fos_text}",
        xaxis_title="Distance from downstream toe (m)",
        yaxis_title="Elevation (m)",
        height=420,
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Main rendering function (imported by geotechnical_dashboard.py)
# ═══════════════════════════════════════════════════════════════════════════

def render_slope_stability_tab():
    """
    Renders the complete "Slope Stability & Erosion" tab content inside
    the Geotechnical Workstation.  Expects to be called within an active
    st.tabs() context (i.e. inside ``with tab_slope_stability:``).
    """
    _section_header("Slope Stability & Erosion Analysis")
    st.caption(
        "Bishop's Simplified (moment equilibrium) and Janbu's Simplified "
        "(horizontal force equilibrium) grid search for the critical slip "
        "circle, plus rapid-drawdown, piping-erosion, and wave-overtopping "
        "screening. Underlying numerics: calculator/slope_stability_engine.py."
    )

    # ------------------------------------------------------------------
    # Sidebar inputs -- keyed to avoid conflicts with the parent dashboard
    # ------------------------------------------------------------------
    with st.sidebar:
        st.markdown("---")
        st.header("Slope Stability Params")

        ss_dam_h = st.number_input(
            "Dam height (m)", value=20.0, min_value=3.0, max_value=100.0,
            step=1.0, key="ss_dam_h",
        )
        ss_water = st.number_input(
            "Reservoir depth (m)", value=18.0, min_value=1.0,
            max_value=float(ss_dam_h), step=1.0, key="ss_water",
        )
        ss_mu = st.number_input(
            "Upstream slope H:1V", value=3.0, min_value=1.0,
            max_value=6.0, step=0.5, key="ss_mu",
        )
        ss_md = st.number_input(
            "Downstream slope H:1V", value=2.0, min_value=1.0,
            max_value=6.0, step=0.5, key="ss_md",
        )
        ss_crest = st.number_input(
            "Crest width (m)", value=6.0, min_value=2.0,
            max_value=20.0, step=1.0, key="ss_crest",
        )

        st.subheader("Soil Strength")
        ss_c = st.number_input(
            "Effective cohesion c' (kPa)", value=15.0, min_value=0.0,
            max_value=200.0, step=1.0, key="ss_c",
        )
        ss_phi = st.number_input(
            "Effective friction φ' (°)", value=28.0, min_value=0.0,
            max_value=50.0, step=1.0, key="ss_phi",
        )
        ss_gamma = st.number_input(
            "Soil unit weight γ (kN/m³)", value=19.0, min_value=14.0,
            max_value=26.0, step=0.5, key="ss_gamma",
        )

    # Build the engine
    engine = AdvancedSlopeStabilityEngine(
        ss_dam_h, ss_water, ss_mu, ss_md, ss_crest,
        cohesion_kpa=ss_c,
        friction_angle_deg=ss_phi,
        soil_unit_weight_kn_m3=ss_gamma,
    )

    # Phreatic line for the plots
    phreatic = phreatic_line_no_filter(
        ss_dam_h, ss_water, ss_mu, ss_md, ss_crest,
    )

    # ==================================================================
    # Sub-tabs inside the Slope Stability tab
    # ==================================================================
    sub_bishop, sub_drawdown, sub_piping, sub_overtopping = st.tabs([
        "🔺 Bishop / Janbu",
        "🌊 Rapid Drawdown",
        "🕳️ Progressive Piping",
        "💧 Wave Overtopping",
    ])

    # ------------------------------------------------------------------
    # 1. Bishop & Janbu
    # ------------------------------------------------------------------
    with sub_bishop:
        _section_header("Bishop's Simplified vs Janbu's Simplified")
        st.caption(
            "Two independent equilibrium formulations on the same grid of "
            "trial toe circles — if they agree closely, the result is more "
            "trustworthy than either alone."
        )

        n_centers = st.slider(
            "Grid resolution (n × n centers)", 8, 20, 12,
            key="ss_ncenters",
            help="Higher = finer search grid, slower computation.",
        )

        if st.button("Run slope-stability search", key="ss_run_bishop"):
            with st.spinner("Searching trial circles (Bishop + Janbu)…"):
                results = engine.multi_method_slope_stability(n_centers=n_centers)

            bishop = results["bishop_simplified"]
            janbu = results["janbu_simplified"]

            col_b, col_j = st.columns(2)
            with col_b:
                fos_b = bishop["fos"]
                st.metric("Bishop FoS (min)", f"{fos_b:.2f}" if np.isfinite(fos_b) else "N/A")
                _pass_fail_badge(
                    f"Bishop FoS = {fos_b:.2f}" if np.isfinite(fos_b) else "No valid circle found",
                    np.isfinite(fos_b) and fos_b >= 1.5,
                )
            with col_j:
                fos_j = janbu["fos"]
                st.metric("Janbu FoS (min)", f"{fos_j:.2f}" if np.isfinite(fos_j) else "N/A")
                _pass_fail_badge(
                    f"Janbu FoS = {fos_j:.2f}" if np.isfinite(fos_j) else "No valid circle found",
                    np.isfinite(fos_j) and fos_j >= 1.5,
                )

            # Cross-section plots
            fig_b = _plot_cross_section_with_circle(
                ss_dam_h, ss_mu, ss_md, ss_crest, ss_water,
                bishop, "Bishop's Simplified", phreatic,
            )
            st.plotly_chart(fig_b, use_container_width=True)

            fig_j = _plot_cross_section_with_circle(
                ss_dam_h, ss_mu, ss_md, ss_crest, ss_water,
                janbu, "Janbu's Simplified", phreatic,
            )
            st.plotly_chart(fig_j, use_container_width=True)

            # Agreement check
            if np.isfinite(fos_b) and np.isfinite(fos_j):
                diff_pct = abs(fos_b - fos_j) / max(fos_b, fos_j) * 100
                _glass_card(
                    f"Bishop = <b>{fos_b:.2f}</b> &nbsp;|&nbsp; "
                    f"Janbu = <b>{fos_j:.2f}</b> &nbsp;|&nbsp; "
                    f"Difference = <b>{diff_pct:.1f}%</b>"
                    + (" ✓ (good agreement)" if diff_pct < 15 else
                       " ⚠️ large discrepancy — refine grid or check inputs")
                )

    # ------------------------------------------------------------------
    # 2. Rapid Drawdown
    # ------------------------------------------------------------------
    with sub_drawdown:
        _section_header("Simplified Rapid-Drawdown Check")
        st.caption(
            "Uses an ru-based pore-pressure override (not the full USACE "
            "three-stage method) to represent pore pressures that lag behind "
            "a rapidly falling reservoir."
        )

        ru_val = st.slider(
            "Pore-pressure ratio ru", 0.0, 0.8, 0.4, step=0.05,
            key="ss_ru",
            help="Typical ru ≈ 0.3–0.5 for rapid, complete drawdown of a "
                 "compacted clay core.",
        )

        if st.button("Run rapid-drawdown analysis", key="ss_run_drawdown"):
            with st.spinner("Searching (rapid-drawdown pore pressures)…"):
                dd_result = engine.rapid_drawdown_analysis(ru=ru_val, n_centers=12)

            fos_dd = dd_result["fos"]
            st.metric("Drawdown FoS (Bishop w/ ru)", f"{fos_dd:.2f}" if np.isfinite(fos_dd) else "N/A")
            _pass_fail_badge(
                f"Drawdown FoS = {fos_dd:.2f}" if np.isfinite(fos_dd) else "No valid circle found",
                np.isfinite(fos_dd) and fos_dd >= 1.3,
            )

            if dd_result["slices"] is not None:
                fig_dd = _plot_cross_section_with_circle(
                    ss_dam_h, ss_mu, ss_md, ss_crest, ss_water,
                    dd_result, f"Rapid Drawdown (ru={ru_val})", phreatic,
                )
                st.plotly_chart(fig_dd, use_container_width=True)

            _glass_card(
                f"ru = <b>{ru_val}</b> &nbsp;→&nbsp; FoS = <b>"
                f"{fos_dd:.2f}</b>" if np.isfinite(fos_dd) else
                f"ru = <b>{ru_val}</b> — no valid slip circle found at this geometry."
            )
            st.info(
                "⚠️ This is the **simplified** ru-based approach. For critical "
                "structures, use the full USACE three-stage undrained-strength method.",
                icon="ℹ️",
            )

    # ------------------------------------------------------------------
    # 3. Progressive Piping Erosion
    # ------------------------------------------------------------------
    with sub_piping:
        _section_header("Progressive Piping — Hanson Erosion at Seepage Exit")
        st.caption(
            "Hanson & Simon (2001) excess-shear-stress erosion applied at the "
            "downstream exit face. Requires lab-measured kd and τc (JET / HET) "
            "— no defaults are fabricated."
        )

        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            pp_exit_grad = st.number_input(
                "Exit gradient (i)", value=0.3, min_value=0.0,
                max_value=5.0, step=0.05, key="ss_exit_grad",
                help="From the Piping FoS tab, or your own seepage model.",
            )
        with col_p2:
            pp_tau_c = st.number_input(
                "Critical shear τc (Pa)", value=2.0, min_value=0.0,
                max_value=100.0, step=0.5, key="ss_pp_tauc",
                help="Lab-measured from JET or HET — soil-specific.",
            )
        with col_p3:
            pp_kd = st.number_input(
                "Erodibility kd (cm³/N·s)", value=2.0, min_value=0.01,
                max_value=100.0, step=0.5, key="ss_pp_kd",
                help="Lab-measured — order-of-magnitude scatter is normal.",
            )

        pp_depth = st.number_input(
            "Assumed seepage flow depth at exit (m)", value=0.02,
            min_value=0.001, max_value=1.0, step=0.005,
            format="%.3f", key="ss_pp_depth",
            help="Typically mm-to-cm scale for an initial piping seep.",
        )

        if st.button("Compute piping erosion rate", key="ss_run_piping"):
            piping = engine.progressive_piping_erosion_rate(
                exit_gradient=pp_exit_grad,
                tau_c_pa=pp_tau_c,
                kd_cm3_per_N_s=pp_kd,
                seepage_flow_depth_m=pp_depth,
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Applied shear τ (Pa)", f"{piping['applied_shear_stress_pa']:.2f}")
            c2.metric("Erosion rate", f"{piping['erosion_rate_mm_per_hour']:.3f} mm/hr")
            c3.metric("Critical shear τc (Pa)", f"{piping['critical_shear_stress_pa']:.2f}")

            eroding = piping["applied_shear_stress_pa"] > piping["critical_shear_stress_pa"]
            _pass_fail_badge(
                "EROSION ACTIVE — τ > τc" if eroding else "No erosion — τ ≤ τc (safe)",
                not eroding,
            )

            _glass_card(
                f"Exit gradient = <b>{piping['exit_gradient']:.3f}</b> &nbsp;|&nbsp; "
                f"Assumed depth = <b>{piping['assumed_seepage_flow_depth_m']*100:.1f} cm</b><br>"
                f"Applied τ = <b>{piping['applied_shear_stress_pa']:.2f} Pa</b> &nbsp;→&nbsp; "
                f"Erosion rate = <b>{piping['erosion_rate_mm_per_hour']:.3f} mm/hr</b>"
            )

    # ------------------------------------------------------------------
    # 4. Wave Overtopping Erosion
    # ------------------------------------------------------------------
    with sub_overtopping:
        _section_header("Wave-Overtopping Crest & Face Erosion")
        st.caption(
            "Broad-crested weir discharge → Manning normal-flow depth on the "
            "downstream face → bed shear stress → Hanson erosion rate."
        )

        col_o1, col_o2 = st.columns(2)
        with col_o1:
            ot_head = st.number_input(
                "Head above crest (m)", value=0.3, min_value=0.0,
                max_value=5.0, step=0.1, key="ss_ot_head",
            )
            ot_manning = st.number_input(
                "Manning's n (downstream face)", value=0.035,
                min_value=0.01, max_value=0.10, step=0.005,
                format="%.3f", key="ss_ot_n",
            )
        with col_o2:
            ot_tau_c = st.number_input(
                "Critical shear τc (Pa)", value=5.0, min_value=0.0,
                max_value=200.0, step=1.0, key="ss_ot_tauc",
            )
            ot_kd = st.number_input(
                "Erodibility kd (cm³/N·s)", value=2.0, min_value=0.01,
                max_value=100.0, step=0.5, key="ss_ot_kd",
            )

        if st.button("Compute overtopping erosion", key="ss_run_overtop"):
            ot_result = engine.wave_overtopping_crest_erosion(
                head_above_crest_m=ot_head,
                tau_c_pa=ot_tau_c,
                kd_cm3_per_N_s=ot_kd,
                manning_n=ot_manning,
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Unit discharge q", f"{ot_result['overtopping_unit_discharge_m2_s']:.4f} m²/s")
            c2.metric("Flow depth on face", f"{ot_result['flow_depth_on_face_m']*100:.1f} cm")
            c3.metric("Face shear τ (Pa)", f"{ot_result['applied_shear_stress_pa']:.2f}")

            st.metric("Erosion rate", f"{ot_result['erosion_rate_mm_per_hour']:.3f} mm/hr")

            eroding = ot_result["applied_shear_stress_pa"] > ot_result["critical_shear_stress_pa"]
            _pass_fail_badge(
                "EROSION ACTIVE — τ > τc" if eroding else "No erosion — τ ≤ τc (safe)",
                not eroding,
            )

            _glass_card(
                f"Head above crest = <b>{ot_head:.2f} m</b> &nbsp;→&nbsp; "
                f"q = <b>{ot_result['overtopping_unit_discharge_m2_s']:.4f} m²/s</b><br>"
                f"Downstream face τ = <b>{ot_result['applied_shear_stress_pa']:.2f} Pa</b> "
                f"(τc = {ot_result['critical_shear_stress_pa']:.1f} Pa) &nbsp;→&nbsp; "
                f"Erosion = <b>{ot_result['erosion_rate_mm_per_hour']:.3f} mm/hr</b>"
            )
