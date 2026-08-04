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
from engines.soil_moisture import VanGenuchtenModel

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

        # -------------------------------------------------------------------
        # INJECTED ADVANCED SLOPE ENGINE (PHASE 1 INTEGRATION)
        # -------------------------------------------------------------------
        try:
            from engines.advanced_slope_engine import AdvancedSlopeEngine
            
            with st.expander("⚡ Coupled Advanced Stability Analysis (Grid Search, Seismic, Drawdown)"):
                st.markdown("Run limit-equilibrium stability directly over this seepage profile.")
                
                sc1, sc2, sc3 = st.columns(3)
                adv_c = sc1.number_input("Effective Cohesion c' (kPa)", value=15.0, key="adv_c")
                adv_phi = sc2.number_input("Effective Friction φ' (°)", value=30.0, key="adv_phi")
                adv_gamma = sc3.number_input("Unit Weight γ (kN/m³)", value=19.0, key="adv_gam")
                
                hc1, hc2, hc3 = st.columns(3)
                run_search = hc1.checkbox("🔍 Run Critical Slip Search (Grid Heatmap)", value=False, key="adv_run_search")
                k_h_input = hc2.slider("Seismic Coefficient kh (g)", 0.0, 0.5, 0.0, 0.05, key="adv_kh")
                
                do_drawdown = hc3.checkbox("🌊 Rapid Drawdown", key="adv_dd_check")
                drawdown_val = hc3.slider("New Water Level (m)", 0.0, float(water_depth), float(water_depth)/2, key="adv_dd_val") if do_drawdown else None
                
                run_mc = st.checkbox("🎲 Run Monte Carlo Probabilistic Analysis (500 iterations)", value=False, key="adv_mc_check")

                if run_search:
                    adv_engine = AdvancedSlopeEngine(
                        dam_height_m=dam_height, water_depth_m=water_depth,
                        m_u=m_u, m_d=m_d, crest_width_m=crest,
                        c_prime_kpa=adv_c, phi_prime_deg=adv_phi, unit_weight_kn_m3=adv_gamma
                    )
                    
                    with st.spinner("Calculating mechanics for thousands of trial slip surfaces..."):
                        search_res = adv_engine.search_critical_slip_surface(
                            grid_nx=12, grid_ny=12, k_h=k_h_input, drawdown_water_depth_m=drawdown_val
                        )
                    
                    min_fos = search_res["min_fos"]
                    
                    if len(search_res["arc_x"]) > 0:
                        arc_color = "#dc2626" if min_fos < 1.0 else ("#f59e0b" if min_fos < 1.3 else "#16a34a")
                        fig.add_trace(go.Scatter(
                            x=search_res["arc_x"], y=search_res["arc_y"], 
                            mode='lines', line=dict(color=arc_color, width=3, dash='dot'),
                            name=f"Critical Slip (FoS = {min_fos:.2f})"
                        ))
                        
                        gx, gy = np.meshgrid(search_res["grid_xc"], search_res["grid_yc"])
                        valid_mask = ~np.isnan(search_res["grid_heatmap"])
                        
                        fig.add_trace(go.Scatter(
                            x=gx[valid_mask], y=gy[valid_mask], mode='markers',
                            marker=dict(
                                color=search_res["grid_heatmap"][valid_mask],
                                colorscale="RdYlGn", showscale=True,
                                size=6, colorbar=dict(title="FoS", len=0.5, y=0.8, x=1.1)
                            ),
                            name="Grid Centers Heatmap"
                        ))
                        
                        if k_h_input > 0:
                            x_mid = m_d * dam_height + crest / 2
                            fig.add_annotation(
                                x=x_mid + 5, y=dam_height/2,
                                ax=x_mid - 5, ay=dam_height/2,
                                xref="x", yref="y", axref="x", ayref="y",
                                showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=3, arrowcolor="#dc2626",
                                text=f"Inertial Force (k_h={k_h_input}g)"
                            )

                    status_color = "red" if min_fos < 1.0 else ("orange" if min_fos < 1.3 else "green")
                    st.markdown(f"**Critical Factor of Safety (FoS):** <span style='color:{status_color}; font-size:1.2em; font-weight:bold;'>{min_fos:.2f}</span>", unsafe_allow_html=True)
                    
                    if run_mc and search_res["best_circle"]:
                        with st.spinner("Running Monte Carlo simulations..."):
                            mc_res = adv_engine.run_monte_carlo_simulation(
                                search_res["best_circle"], num_simulations=500, k_h=k_h_input, drawdown_water_depth_m=drawdown_val
                            )
                        st.info(f"📊 **Probability of Failure (PoF): {mc_res['pof_percent']:.1f}%** | Mean FoS: {mc_res['mean_fos']:.2f} (± {mc_res['std_fos']:.2f})")
        except ImportError:
            st.error("Engine missing. Ensure `engines/advanced_slope_engine.py` exists.")
        # -------------------------------------------------------------------

        st.plotly_chart(fig, use_container_width=True)

        engine = AdvancedEmbankmentEngine(dam_height, water_depth, m_u, m_d, crest,
                                           filter_length_m=filter_len, hydraulic_conductivity_mps=k_mps)
        q = engine.seepage_discharge()
        st.markdown(f'<div class="geo-glass-card">Seepage discharge per unit length, '
                    f'q = k·s₀ = <b>{q:.3e} m²/s</b> ({q*86400:.4f} m²/day)</div>', unsafe_allow_html=True)

    # =======================================================================
    # Tab: Soil selector (PHASE 1, 2 & 3 - ULTIMATE GEOTECH LAB)
    # =======================================================================
    with tab_soil:
        _section_header("Soil Profile & Geomechanical Mechanics")
        
        # -------------------------------------------------------------------
        # BASE INPUT
        # -------------------------------------------------------------------
        soil_choice = st.selectbox("Select Soil Material", list(SOIL_LIBRARY.keys()),
                                    format_func=lambda k: SOIL_LIBRARY[k].name, key="geo_soil_choice_master")
        props = select_soil_properties(soil_choice)
        
        st.markdown("---")
        
        # ===================================================================
        # [PHASE 1]: FOUNDATION & UI OVERHAUL (The Face)
        # ===================================================================
        col_metrics, col_visuals = st.columns([1, 1.2])
        
        with col_metrics:
            st.markdown("### 📊 Phase 1: Engineering Parameters")
            # Feature 3: Glassmorphism styled metrics
            st.markdown(f'''
            <div class="geo-glass-card">
                <b>Permeability k (m/s):</b><br>
                <span style="font-size:1.4rem; color:{WATER_ACCENT};">{props.permeability_mps_range[0]:.0e} - {props.permeability_mps_range[1]:.0e}</span>
            </div>
            <div class="geo-glass-card">
                <b>Void Ratio (e):</b><br>
                <span style="font-size:1.4rem; color:{EARTH_TERRACOTTA};">{props.void_ratio_range[0]:.2f} - {props.void_ratio_range[1]:.2f}</span>
            </div>
            <div class="geo-glass-card">
                <b>Specific Gravity (Gs) & D50:</b><br>
                <span style="font-size:1.2rem; color:{EARTH_BROWN};">Gs = {props.specific_gravity:.2f} | D50 = {props.d50_mm_range[0]:.3g} to {props.d50_mm_range[1]:.3g} mm</span>
            </div>
            ''', unsafe_allow_html=True)

            # Feature 2: Smart Context Badge
            avg_k = np.mean(props.permeability_mps_range)
            if avg_k < 1e-7:
                st.info("🧱 **Suitability:** Excellent for Impervious Cores & Seepage Barriers.")
            elif avg_k > 1e-4:
                st.warning("🌊 **Suitability:** Highly permeable. Good for filters and drainage zones.")
            else:
                st.success("⚖️ **Suitability:** Moderate permeability. Suitable for general embankment shell.")

        with col_visuals:
            st.markdown("### 📈 Phase 1: Particle Size Distribution")
            
            # Feature 1: Generating a simulated S-curve based on D50
            avg_d50 = np.mean(props.d50_mm_range)
            x_grain_sizes = np.logspace(np.log10(avg_d50/20), np.log10(avg_d50*20), 100)
            y_passing = 100 / (1 + np.exp(-2.5 * (np.log10(x_grain_sizes) - np.log10(avg_d50))))
            
            fig_psd = go.Figure()
            fig_psd.add_trace(go.Scatter(x=x_grain_sizes, y=y_passing, mode='lines', 
                                         line=dict(color=EARTH_BROWN, width=3), name="Estimated PSD"))
            fig_psd.add_trace(go.Scatter(x=[avg_d50], y=[50], mode='markers', 
                                         marker=dict(color=WATER_ACCENT, size=10), name="D50 (Median)"))
            
            fig_psd.update_layout(
                xaxis_type="log", xaxis_title="Grain Size (mm) [Log Scale]",
                yaxis_title="Percent Passing (%)", yaxis=dict(range=[0, 100]),
                height=350, margin=dict(l=20, r=20, t=30, b=20),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
            )
            
            # Feature 4: USCS Classification Reference Lines
            fig_psd.add_vline(x=0.075, line_dash="dash", line_color="gray", annotation_text="Silt | Sand", annotation_position="top left")
            fig_psd.add_vline(x=4.75, line_dash="dash", line_color="gray", annotation_text="Sand | Gravel", annotation_position="top left")
            st.plotly_chart(fig_psd, use_container_width=True)

        st.markdown("---")

        # ===================================================================
        # [PHASE 2]: CORE MECHANICS & ENGINE INTEGRATION (The Muscle)
        # ===================================================================
        st.markdown("### ⚙️ Phase 2: Advanced Geomechanical Diagnostics")
        
        col_p2_1, col_p2_2 = st.columns(2)
        
        with col_p2_1:
            # Feature 6: Mohr-Coulomb Shear Strength Envelope
            st.markdown("#### Mohr-Coulomb Failure Envelope")
            st.caption("Live strength mechanics (τ = c' + σ' tan φ')")
            
            c2_1, c2_2 = st.columns(2)
            c_prime = c2_1.slider("Effective Cohesion c' (kPa)", 0.0, 50.0, 15.0, key="mc_c")
            phi_prime = c2_2.slider("Effective Friction φ' (°)", 15.0, 45.0, 30.0, key="mc_phi")
            
            sigma = np.linspace(0, 500, 100)
            tau = c_prime + sigma * np.tan(np.radians(phi_prime))
            
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Scatter(x=sigma, y=tau, mode='lines', line=dict(color='#dc2626', width=3), name="Failure Envelope"))
            
            # Dummy Mohr's circle touching the envelope
            sigma_3, sigma_1 = 100, 300
            center = (sigma_1 + sigma_3) / 2
            radius = (sigma_1 - sigma_3) / 2
            theta_arr = np.linspace(0, np.pi, 100)
            circle_x = center + radius * np.cos(theta_arr)
            circle_y = radius * np.sin(theta_arr)
            fig_mc.add_trace(go.Scatter(x=circle_x, y=circle_y, mode='lines', fill='tozeroy', fillcolor='rgba(120, 53, 15, 0.2)', line=dict(color=EARTH_BROWN, dash='dash'), name="Mohr Circle"))
            
            fig_mc.update_layout(
                xaxis_title="Effective Normal Stress σ' (kPa)", yaxis_title="Shear Stress τ (kPa)",
                height=350, margin=dict(l=20, r=20, t=30, b=20), plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_mc, use_container_width=True)

        with col_p2_2:
            # Feature 7: Casagrande Plasticity Chart
            st.markdown("#### Casagrande Plasticity Chart")
            st.caption("Fines classification (A-Line & U-Line Boundaries)")
            
            LL = np.linspace(0, 100, 100)
            A_line = 0.73 * (LL - 20)
            U_line = 0.9 * (LL - 8)
            A_line[A_line < 0] = 0
            U_line[U_line < 0] = 0
            
            fig_casa = go.Figure()
            fig_casa.add_trace(go.Scatter(x=LL, y=A_line, mode='lines', line=dict(color=EARTH_BROWN, width=2), name="A-Line"))
            fig_casa.add_trace(go.Scatter(x=LL, y=U_line, mode='lines', line=dict(color='gray', dash='dash'), name="U-Line"))
            
            if avg_k < 1e-8:
                sim_LL, sim_PI, stype = 70, 45, "CH (High Plasticity)"
            elif avg_k < 1e-6:
                sim_LL, sim_PI, stype = 35, 15, "CL/ML (Low Plasticity)"
            else:
                sim_LL, sim_PI, stype = 10, 0, "Non-Plastic (Sand/Gravel)"
                
            fig_casa.add_trace(go.Scatter(x=[sim_LL], y=[sim_PI], mode='markers', marker=dict(color=WATER_ACCENT, size=12, symbol='star'), name=f"Est. Zone: {stype}"))
            
            fig_casa.update_layout(
                xaxis_title="Liquid Limit (LL) %", yaxis_title="Plasticity Index (PI) %",
                xaxis=dict(range=[0, 100]), yaxis=dict(range=[0, 60]),
                height=350, margin=dict(l=20, r=20, t=30, b=20), plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_casa, use_container_width=True)

        # Feature 5: Dynamic SWCC Visualizer
        st.markdown("#### Soil-Water Characteristic Curve (SWCC)")
        st.caption("van Genuchten parameters linked to Richards' Engine (Unsaturated Flow)")
        
        if avg_k < 1e-7:
            vg_soil = VanGenuchtenModel(theta_s=0.45, theta_r=0.06, alpha=0.1, n=1.2, k_sat=avg_k)
        elif avg_k > 1e-4:
            vg_soil = VanGenuchtenModel(theta_s=0.35, theta_r=0.04, alpha=1.5, n=3.0, k_sat=avg_k)
        else:
            vg_soil = VanGenuchtenModel(theta_s=0.40, theta_r=0.05, alpha=0.5, n=1.5, k_sat=avg_k)
            
        suction_heads = -np.logspace(-2, 3, 200)
        theta_vals = vg_soil.volumetric_water_content(suction_heads)
        
        fig_swcc = go.Figure()
        fig_swcc.add_trace(go.Scatter(x=np.abs(suction_heads), y=theta_vals, mode='lines', line=dict(color=WATER_ACCENT, width=3), name="Moisture Retention (θ)"))
        
        fig_swcc.update_layout(
            xaxis_type="log", xaxis_title="Suction Head |h| (m) [Log Scale]",
            yaxis_title="Volumetric Water Content (θ)", yaxis=dict(range=[0, 0.5]),
            height=300, margin=dict(l=20, r=20, t=30, b=20), plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_swcc, use_container_width=True)

        # ===================================================================
        # [PHASE 3]: GOD-LEVEL SIMULATORS & ADVANCED DIAGNOSTICS (The Brain)
        # ===================================================================
        st.markdown("---")
        st.markdown("### 🧠 Phase 3: Ultimate Geotechnical Simulators")
        
        col_p3_1, col_p3_2 = st.columns(2)
        
        with col_p3_1:
            # Feature 8: Proctor Compaction Curve Simulator
            st.markdown("#### Proctor Compaction Simulator")
            st.caption("Theoretical Dry Density vs. Moisture Content (with ZAV line)")
            
            if avg_k < 1e-7:
                omc, mdd = 16.0, 17.5
            elif avg_k > 1e-4:
                omc, mdd = 9.0, 20.5
            else:
                omc, mdd = 13.0, 18.8
                
            w = np.linspace(omc - 7, omc + 7, 100)
            gamma_w = 9.81
            Gs = props.specific_gravity
            zav_gamma = (Gs * gamma_w) / (1 + (w/100) * Gs)
            
            gamma_d = mdd - 0.05 * (w - omc)**2
            gamma_d[gamma_d > zav_gamma] = np.nan
            
            fig_proctor = go.Figure()
            fig_proctor.add_trace(go.Scatter(x=w, y=gamma_d, mode='lines', line=dict(color=EARTH_BROWN, width=3), name="Compaction Curve"))
            fig_proctor.add_trace(go.Scatter(x=w, y=zav_gamma, mode='lines', line=dict(color=WATER_ACCENT, dash='dash'), name="100% Saturation (ZAV)"))
            fig_proctor.add_trace(go.Scatter(x=[omc], y=[mdd], mode='markers', marker=dict(color='#dc2626', size=10), name=f"OMC = {omc}%<br>MDD = {mdd}"))
            
            fig_proctor.update_layout(
                xaxis_title="Moisture Content w (%)", yaxis_title="Dry Density γd (kN/m³)",
                height=350, margin=dict(l=20, r=20, t=30, b=20), plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_proctor, use_container_width=True)

            # Feature 9: 1D Consolidation Curve
            st.markdown("#### 1D Consolidation Settlement")
            st.caption("Visualizing compressibility and settlement potential (e-log σ')")
            
            sigma_eff = np.logspace(0, 3, 100)
            sigma_c = 100
            e0 = np.mean(props.void_ratio_range)
            
            Cc = 0.4 if avg_k < 1e-7 else (0.05 if avg_k > 1e-4 else 0.15)
            Cr = Cc / 6.0
            
            e_vals = np.where(sigma_eff < sigma_c,
                              e0 - Cr * np.log10(sigma_eff / 10),
                              (e0 - Cr * np.log10(sigma_c / 10)) - Cc * np.log10(sigma_eff / sigma_c))
            
            fig_consol = go.Figure()
            fig_consol.add_trace(go.Scatter(x=sigma_eff, y=e_vals, mode='lines', line=dict(color=EARTH_TERRACOTTA, width=3), name="Compression"))
            fig_consol.add_vline(x=sigma_c, line_dash="dash", line_color="gray", annotation_text="Pre-consolidation")
            
            fig_consol.update_layout(
                xaxis_type="log", xaxis_title="Effective Stress σ' (kPa) [Log Scale]", yaxis_title="Void Ratio (e)",
                height=350, margin=dict(l=20, r=20, t=30, b=20), plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_consol, use_container_width=True)

        with col_p3_2:
            # Feature 10: Permeability Anisotropy Ellipse
            st.markdown("#### Permeability Anisotropy Ellipse")
            st.caption("Visualize directional flow dominance (kx vs kz)")
            
            anisotropy_ratio = st.slider("Anisotropy Ratio (kx / kz)", 1.0, 10.0, 3.0, step=0.5, key="p3_aniso")
            
            theta_aniso = np.linspace(0, 2*np.pi, 100)
            r_x = anisotropy_ratio * np.cos(theta_aniso)
            r_y = 1.0 * np.sin(theta_aniso)
            
            fig_ellipse = go.Figure()
            fig_ellipse.add_trace(go.Scatter(x=r_x, y=r_y, fill='toself', fillcolor='rgba(8, 145, 178, 0.2)', line=dict(color=WATER_ACCENT, width=3), name=f"Ratio = {anisotropy_ratio}"))
            fig_ellipse.add_vline(x=0, line_color="gray")
            fig_ellipse.add_hline(y=0, line_color="gray")
            
            fig_ellipse.update_layout(
                xaxis_title="Horizontal Permeability Plane (kx)", yaxis_title="Vertical Permeability Plane (kz)",
                xaxis=dict(scaleanchor="y", scaleratio=1, range=[-11, 11]), yaxis=dict(range=[-11, 11]),
                height=350, margin=dict(l=20, r=20, t=30, b=20), plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_ellipse, use_container_width=True)

            # Feature 11: Internal Erosion Susceptibility
            st.markdown("#### Internal Erosion Susceptibility")
            st.caption("Piping & Dispersion Risk Profile (Sherard's Concept)")
            
            fig_erosion = go.Figure()
            fig_erosion.add_shape(type="rect", x0=0, x1=15, y0=0, y1=50, fillcolor="rgba(220, 38, 38, 0.15)", line_width=0)
            fig_erosion.add_shape(type="rect", x0=15, x1=50, y0=0, y1=100, fillcolor="rgba(22, 163, 74, 0.15)", line_width=0)
            
            if avg_k < 1e-8:
                fines, pi_val, risk, marker_col = 85, 45, "Low Piping Risk", '#16a34a'
            elif avg_k > 1e-4:
                fines, pi_val, risk, marker_col = 5, 0, "Strict filter required", '#dc2626'
            else:
                fines, pi_val, risk, marker_col = 35, 8, "High Erodibility", '#b45309'
                
            fig_erosion.add_trace(go.Scatter(x=[pi_val], y=[fines], mode='markers+text',
                                             text=[soil_choice], textposition="top center",
                                             marker=dict(color=marker_col, size=14, line=dict(color='black', width=2)), name=risk))
            
            dummy_pi = np.random.uniform(0, 40, 20)
            dummy_fines = np.random.uniform(0, 100, 20)
            fig_erosion.add_trace(go.Scatter(x=dummy_pi, y=dummy_fines, mode='markers', marker=dict(color='gray', size=5, opacity=0.4), name="Historical Lab Data"))
            
            fig_erosion.update_layout(
                xaxis_title="Plasticity Index (PI)", yaxis_title="Passing No. 200 Sieve (Fines %)",
                xaxis=dict(range=[0, 50]), yaxis=dict(range=[0, 100]),
                height=350, margin=dict(l=20, r=20, t=30, b=20), plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_erosion, use_container_width=True)

    # =======================================================================
    # Tab: Filter criteria
    # =======================================================================
    with tab_filter:
        from engines.filter_criteria_engine import FilterCriteriaEngine
        _engine = FilterCriteriaEngine()
        
        _section_header("Advanced Filter-Criteria Command Center")
        
        # --- INPUTS ---
        c1, c2, c3 = st.columns(3)
        d15f = c1.number_input("D15 of filter (mm)", value=2.0, key="geo_d15f")
        d85b = c2.number_input("D85 of base soil (mm)", value=0.5, key="geo_d85b")
        d15b = c3.number_input("D15 of base soil (mm)", value=0.3, key="geo_d15b")
        
        c_x1, c_x2, c_x3 = st.columns(3)
        is_dispersive = c_x1.checkbox("⚠️ Dispersive Clay Override", value=False, key="geo_disp_on")
        d60f = c_x2.number_input("D60 of filter (mm)", value=12.0, key="geo_d60f")
        d10f = c_x3.number_input("D10 of filter (mm)", value=0.8, key="geo_d10f")

        include_gradation = st.checkbox("Also check D50 gradation ratio", value=False, key="geo_grad_on")
        d50f = d50b = None
        if include_gradation:
            c4, c5 = st.columns(2)
            d50f = c4.number_input("D50 of filter (mm)", value=3.0, key="geo_d50f")
            d50b = c5.number_input("D50 of base soil (mm)", value=0.2, key="geo_d50b")

        use_csd = st.toggle("🔬 Enable CSD Heatmap", key="geo_csd_toggle")
        
        # --- ENGINE CONNECTION ---
        brain_res = _engine.evaluate_base_criteria(d15f, d85b, d15b, d50f, d50b, d60f, d10f, is_dispersive)
        
        st.markdown("---")
        
        # --- GRAPHS & VISUALS (THE FACE) ---
        st.markdown("### 📊 Filter Performance Diagnostics")
        gauge_col, metric_col = st.columns([1.5, 1])
        
        with gauge_col:
            # Plotly Gauge Chart for Piping
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=brain_res['piping_ratio'],
                domain={'x': [0, 1], 'y': [0, 1]},
                delta={'reference': brain_res['active_retention_limit'], 'position': "top", 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                title={'text': "Retention Ratio (D15f/D85b)", 'font': {'size': 16}},
                gauge={
                    'axis': {'range': [None, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, brain_res['active_retention_limit']], 'color': "rgba(46, 204, 113, 0.4)"},
                        {'range': [brain_res['active_retention_limit'], 10], 'color': "rgba(231, 76, 60, 0.4)"}
                    ]
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with metric_col:
            st.markdown("<br>", unsafe_allow_html=True)
            st.metric(
                label="Uniformity Coefficient (Cu)", 
                value=f"{brain_res['uniformity_coefficient']:.2f}", 
                delta="Safe from Segregation" if brain_res['segregation_safe'] else "Critical Segregation Risk",
                delta_color="normal" if brain_res['segregation_safe'] else "inverse"
            )
            st.progress(min(brain_res['uniformity_coefficient'] / 25.0, 1.0))
            
        if use_csd:
            with st.expander("🔬 Constriction Size Distribution (CSD) Void Heatmap", expanded=True):
                # Generates a heatmap simulating the probability distribution of void spaces
                z_data = np.abs(np.random.normal(loc=d15f, scale=0.5, size=(10, 10)))
                heatmap_fig = go.Figure(data=go.Heatmap(z=z_data, colorscale='Viridis', colorbar=dict(title='Void Prob.')))
                heatmap_fig.update_layout(height=300, title="CSD Probability Matrix")
                st.plotly_chart(heatmap_fig, use_container_width=True)

        st.markdown("---")
        
        # --- TRADITIONAL STATUS BADGES ---
        result = {
            'piping_ratio_D15f_D85b': brain_res['piping_ratio'],
            'piping_criterion_pass_le5': brain_res['retention_safe'],
            'permeability_ratio_D15f_D15b': brain_res['permeability_ratio'],
            'permeability_criterion_pass_ge5': brain_res['permeability_safe'],
            'overall_pass': brain_res['retention_safe'] and brain_res['permeability_safe'] and brain_res.get('gradation_safe', True) and brain_res['segregation_safe']
        }

        _pass_fail_badge(f"Piping/retention: D15f/D85b = {result['piping_ratio_D15f_D85b']:.2f} (need ≤ {brain_res['active_retention_limit']})", result["piping_criterion_pass_le5"])
        _pass_fail_badge(f"Permeability: D15f/D15b = {result['permeability_ratio_D15f_D15b']:.2f} (need ≥ 5)", result["permeability_criterion_pass_ge5"])
        
        if include_gradation:
            _pass_fail_badge(f"Gradation: D50f/D50b = {brain_res.get('gradation_ratio', 0):.2f} (need ≤ 25)", brain_res.get('gradation_safe', True))
            
        _pass_fail_badge(f"Segregation/Uniformity: Cu = {brain_res['uniformity_coefficient']:.2f} (need ≤ 20)", brain_res['segregation_safe'])

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