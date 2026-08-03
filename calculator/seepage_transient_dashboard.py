import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from engines.soil_moisture import VanGenuchtenModel
from solver.pde_solvers import Richards1D, Richards2D

def render_transient_seepage_tab():
    st.subheader("🌧️ 1D vs 2D Transient Infiltration Dynamics")
    st.caption("Solves Richards' PDE for both 1D and 2D. Rainfall in 2D is localized at the center to visualize lateral flow.")

    c1, c2, c3 = st.columns(3)
    soil_depth = c1.slider("Soil Column Depth Z (m)", 1.0, 10.0, 5.0)
    domain_width = c2.slider("Soil Domain Width X (m) [2D Only]", 2.0, 20.0, 10.0)
    rain_rate = c3.slider("Rainfall Rate (mm/hr)", 0.0, 50.0, 25.0)
    sim_hours = st.slider("Simulation Time (Hours)", 1, 48, 12)

    if st.button("Run PDE Simulation (1D & 2D)", type="primary"):
        with st.spinner("Crunching Jacobian Matrices & Solving 1D + 2D Richards' Equations..."):
            
            # 1. Define Soil Model
            vg_soil = VanGenuchtenModel(theta_s=0.45, theta_r=0.06, alpha=0.5, n=1.4, k_sat=1e-5)
            
            # -------------------------------------------------------------
            # RUN 1D SOLVER
            # -------------------------------------------------------------
            solver_1d = Richards1D(vg_soil, z_max_m=soil_depth, n_nodes=25)
            initial_psi_1d = -solver_1d.z_grid  
            res_1d = solver_1d.simulate(initial_psi_1d, t_max_seconds=sim_hours*3600, rainfall_mm_per_hr=rain_rate)
            
            z_1d = res_1d['z_grid']
            theta_final_1d = res_1d['theta_profiles'][-1]

            # -------------------------------------------------------------
            # RUN 2D SOLVER
            # -------------------------------------------------------------
            solver_2d = Richards2D(vg_soil, x_max_m=domain_width, z_max_m=soil_depth, nx=15, nz=25)
            
            # Initial condition: Hydrostatic (depends only on z)
            Z_grid_2d = np.tile(solver_2d.z_grid, (solver_2d.nx, 1)).T
            initial_psi_2d = -Z_grid_2d
            
            res_2d = solver_2d.simulate(initial_psi_2d, t_max_seconds=sim_hours*3600, rainfall_mm_per_hr=rain_rate)
            
            x_2d = res_2d['x_grid']
            z_2d = res_2d['z_grid']
            theta_final_2d = res_2d['final_theta_2d']
            
            # Extract the center column from 2D results for the Intersection Graph
            center_x_idx = solver_2d.nx // 2
            theta_final_2d_center = theta_final_2d[:, center_x_idx]

            # -------------------------------------------------------------
            # PLOT THE 3 GRAPHS
            # -------------------------------------------------------------
            st.markdown("---")
            
            # GRAPH 1: 1D Depth Profile
            c_g1, c_g2 = st.columns(2)
            with c_g1:
                st.markdown("### Graph 1: 1D Moisture Profile")
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=theta_final_1d, y=z_1d, name="1D Column", line=dict(color='#0891b2', width=3)))
                fig1.update_layout(xaxis_title="Volumetric Water Content (θ)", yaxis_title="Elevation (m)", xaxis=dict(range=[0, 0.5]), height=400)
                st.plotly_chart(fig1, use_container_width=True)

            # GRAPH 2: Intersection Graph (1D vs 2D Comparison)
            with c_g2:
                st.markdown("### Graph 3: 1D vs 2D (Intersection)")
                st.caption("Notice how 2D lateral flow reduces vertical penetration compared to strict 1D.")
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=theta_final_1d, y=z_1d, name="1D Model", line=dict(dash='dash', color='gray', width=2)))
                fig3.add_trace(go.Scatter(x=theta_final_2d_center, y=z_2d, name="2D Model (Center Slice)", line=dict(color='#dc2626', width=3)))
                fig3.update_layout(xaxis_title="Volumetric Water Content (θ)", yaxis_title="Elevation (m)", xaxis=dict(range=[0, 0.5]), height=400)
                st.plotly_chart(fig3, use_container_width=True)

            # GRAPH 3: 2D Heatmap
            st.markdown("### Graph 2: 2D Spatial Moisture Distribution (Heatmap)")
            st.caption(f"Localized rainfall at the center forces water to flow both downwards and sideways.")
            fig2 = go.Figure(data=go.Contour(
                z=theta_final_2d,
                x=x_2d,
                y=z_2d,
                colorscale="YlGnBu",
                colorbar=dict(title="Moisture (θ)")
            ))
            fig2.update_layout(xaxis_title="Distance X (m)", yaxis_title="Elevation Z (m)", height=450)
            st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    render_transient_seepage_tab()