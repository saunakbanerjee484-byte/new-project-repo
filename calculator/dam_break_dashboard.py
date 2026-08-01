"""
dam_break_dashboard.py

Interactive Streamlit panel for engines.dam_break.

Two tabs:
  1. 1D Channel Routing -- classic instantaneous dam-break or an
     embankment/earth-dam breach (Froehlich empirical breach -> broad-
     crested weir outflow), routed downstream via the 1D Saint-Venant
     HLL solver. Water-surface profile animation + downstream
     hydrographs.
  2. 2D / 3D Floodplain Inundation -- the same breach physics, but
     spread across a 2D structured floodplain grid (engines.dam_break.
     SaintVenant2D), shown as a top-down depth heatmap AND as a 3D
     surface plot of the water-surface elevation.

A note on "3D": Saint-Venant is a depth-averaged shallow-water
formulation -- there is no independent 3D (free-surface Navier-Stokes /
VOF) solver here, and building one is a much larger undertaking than a
depth-averaged model. What Tab 2 gives you is the genuinely 2D
depth-averaged simulation, rendered two ways: a 2D top-down heatmap and
a 3D surface plot of that same result. That 3D surface is a
*visualization* of the 2D solve, not a separate 3D physics engine --
flagged clearly in the UI so it isn't mistaken for one.

Run standalone:   streamlit run calculator/dam_break_dashboard.py
Or import render_dam_break_tab() into app.py as a tab:
    from calculator.dam_break_dashboard import render_dam_break_tab
"""

import sys
from pathlib import Path

# This file now lives in calculator/, one level below the project root
# where engines/, geospatial/, etc. live. Add the project root to
# sys.path so `from engines.dam_break import ...` below resolves both
# when this module is imported as calculator.dam_break_dashboard (from
# app.py, run from the project root) and when it's run standalone via
# `streamlit run calculator/dam_break_dashboard.py` (which otherwise
# only puts calculator/ itself on sys.path).
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engines.dam_break import (
    SaintVenant1D, SaintVenant2D, froehlich_breach_params,
    breach_outflow_hydrograph, breach_inflow_function, G,
)

try:
    from geospatial.inundation_map import inundation_extent_stats
except ImportError:
    inundation_extent_stats = None


# =============================================================================
# 1D cached simulation runners
# =============================================================================

@st.cache_data(show_spinner=False)
def _run_instantaneous_1d(L, N, dam_km, h_up, h_down, manning_n, width, t_end_min):
    x = np.linspace(0, L, N)
    bed = np.zeros(N)
    dam_index = int(np.searchsorted(x, dam_km * 1000))
    model = SaintVenant1D.dam_break_ic(
        x, bed, dam_index, h_upstream=h_up, h_downstream=max(h_down, 1e-3),
        manning_n=manning_n, width=width,
    )
    gauge_km = sorted(set([round(dam_km + d, 1) for d in (2, 5, 10, 20) if dam_km + d < L / 1000]))
    gauge_idx = {km: int(np.searchsorted(x, km * 1000)) for km in gauge_km}
    result = model.run(t_end_min * 60, cfl=0.45, record_every=max(t_end_min * 60 / 40, 30),
                        gauge_indices=list(gauge_idx.values()))
    return x, bed, result, gauge_idx


@st.cache_data(show_spinner=False)
def _run_breach_1d(L, N, dam_km, dam_height, reservoir_vol_mcm, reservoir_level,
                    dam_crest, dam_base, failure_mode, h_down, manning_n, width, t_end_min):
    params = froehlich_breach_params(dam_height, reservoir_vol_mcm * 1e6, failure_mode)
    t_b, Q_b, h_res = breach_outflow_hydrograph(
        reservoir_level_m=reservoir_level, dam_crest_m=dam_crest, dam_base_m=dam_base,
        breach_params=params, t_end_s=t_end_min * 60,
        storage_area_m2=reservoir_vol_mcm * 1e6 / max(reservoir_level - dam_base, 1.0),
    )
    inflow_fn = breach_inflow_function(t_b, Q_b)

    x = np.linspace(0, L, N)
    bed = np.zeros(N)
    dam_index = int(np.searchsorted(x, dam_km * 1000))
    model = SaintVenant1D(x, bed, manning_n=manning_n, width=width)
    model.set_initial_condition(np.full(N, max(h_down, 1e-3)))

    gauge_km = sorted(set([round(dam_km + d, 1) for d in (2, 5, 10, 20) if dam_km + d < L / 1000]))
    gauge_idx = {km: int(np.searchsorted(x, km * 1000)) for km in gauge_km}
    result = model.run(t_end_min * 60, cfl=0.45, record_every=max(t_end_min * 60 / 40, 30),
                        gauge_indices=list(gauge_idx.values()), inflow_Q=inflow_fn)
    return x, bed, result, gauge_idx, params, (t_b, Q_b)


# =============================================================================
# 2D cached simulation runners
# =============================================================================

@st.cache_data(show_spinner=False)
def _run_2d_instantaneous(nx, ny, dx, reservoir_depth, floodplain_depth,
                           reservoir_extent_km, breach_width_m, manning_n,
                           t_end_min, record_every_s):
    """
    A block of water (representing floodwater that has already surged
    through a breach) released at one edge of an open floodplain grid,
    spreading via the unmodified 2D solver -- the direct 2D analogue of
    the classic instantaneous dam-break.
    """
    model = SaintVenant2D(nx, ny, dx, dx, manning_n=manning_n)
    h0 = np.full((ny, nx), max(floodplain_depth, 1e-3))

    n_x_cells = max(1, int(round(reservoir_extent_km * 1000 / dx)))
    n_breach_cells = max(1, int(round(breach_width_m / dx)))
    y_mid = ny // 2
    y0 = max(0, y_mid - n_breach_cells // 2)
    y1 = min(ny, y0 + n_breach_cells)
    h0[y0:y1, :n_x_cells] = reservoir_depth

    model.set_initial_condition(h0)
    result = model.run(t_end_min * 60, cfl=0.4, record_every=record_every_s)
    return result, (y0, y1)


@st.cache_data(show_spinner=False)
def _run_2d_progressive_breach(nx, ny, dx, dam_height, reservoir_vol_mcm,
                                reservoir_level, dam_base, failure_mode,
                                breach_width_m, floodplain_depth, manning_n,
                                t_end_min, record_every_s):
    """
    Couples the 1D Froehlich breach-outflow hydrograph (engines.dam_break.
    breach_outflow_hydrograph) into the 2D floodplain solver as a growing
    volumetric source injected along a short reach of the inflow edge --
    a more physically representative embankment breach than an
    instantaneous block release, since the breach actually widens over
    its formation time (tf_hours) rather than dumping its full outflow
    at t=0.

    Implementation note: SaintVenant2D.run() has no source-term hook, so
    this steps the solver manually using its own sweep/friction/CFL
    methods (same techniques run() uses internally) and adds the source
    between sub-steps.
    """
    dam_crest = dam_base + dam_height
    params = froehlich_breach_params(dam_height, reservoir_vol_mcm * 1e6, failure_mode)
    t_b, Q_b, _ = breach_outflow_hydrograph(
        reservoir_level_m=reservoir_level, dam_crest_m=dam_crest, dam_base_m=dam_base,
        breach_params=params, t_end_s=t_end_min * 60,
        storage_area_m2=reservoir_vol_mcm * 1e6 / max(reservoir_level - dam_base, 1.0),
    )
    inflow_fn = breach_inflow_function(t_b, Q_b)

    model = SaintVenant2D(nx, ny, dx, dx, manning_n=manning_n)
    model.set_initial_condition(np.full((ny, nx), max(floodplain_depth, 1e-3)))

    n_breach_cells = max(1, int(round(breach_width_m / dx)))
    y_mid = ny // 2
    y0 = max(0, y_mid - n_breach_cells // 2)
    y1 = min(ny, y0 + n_breach_cells)
    cell_area = dx * dx

    h, hu, hv = model.h.copy(), model.hu.copy(), model.hv.copy()
    t, t_end_s, next_record = 0.0, t_end_min * 60, 0.0
    rec_t, rec_h = [], []

    while t < t_end_s:
        dt = min(model._cfl_dt(h, hu, hv, 0.4), 5.0, t_end_s - t)
        Qt = inflow_fn(t)
        h[y0:y1, 0] += Qt * dt / (cell_area * (y1 - y0))

        h, hu, hv = model._sweep_x(h, hu, hv, dt / 2)
        h, hu, hv = model._sweep_y(h, hu, hv, dt)
        h, hu, hv = model._sweep_x(h, hu, hv, dt / 2)
        hu, hv = model._friction(h, hu, hv, dt)
        h = np.maximum(h, 0.0)
        t += dt

        if t >= next_record - 1e-9:
            rec_t.append(t)
            rec_h.append(h.copy())
            next_record += record_every_s

    result = {"t": np.array(rec_t), "h": np.array(rec_h)}
    return result, (y0, y1), params, (t_b, Q_b)


# =============================================================================
# Tab 1: 1D channel routing
# =============================================================================

def _render_1d_tab():
    st.caption("1D Saint-Venant (HLL finite-volume) flood-wave routing along a channel reach.")

    scenario = st.radio(
        "Scenario", ["Instantaneous dam-break (classic)", "Embankment / earth-dam breach"],
        horizontal=True, key="scenario_1d",
    )

    with st.sidebar:
        st.header("1D Channel")
        L_km = st.slider("Reach length (km)", 5, 100, 30, key="L_km_1d")
        dam_km = st.slider("Dam/barrage chainage (km)", 0.0, float(L_km), 2.0, key="dam_km_1d")
        N = st.select_slider("Grid resolution (cells)", [200, 400, 800, 1200], value=400, key="N_1d")
        manning_n = st.slider("Manning's n", 0.020, 0.060, 0.035, key="n_1d")
        width = st.slider("Channel width (m)", 20, 500, 120, key="width_1d")
        h_down = st.slider("Downstream initial depth (m)", 0.0, 5.0, 1.0, key="hdown_1d")
        t_end_min = st.slider("Simulation duration (min)", 30, 360, 120, key="tend_1d")

        st.header("Reservoir / Dam")
        if scenario.startswith("Instantaneous"):
            h_up = st.slider("Upstream reservoir depth (m)", 2.0, 25.0, 12.0, key="hup_1d")
        else:
            dam_height = st.slider("Dam height (m)", 5.0, 60.0, 20.0, key="dh_1d")
            reservoir_vol_mcm = st.slider("Reservoir storage at failure (Mm^3)", 1.0, 500.0, 50.0, key="rv_1d")
            dam_base = st.number_input("Dam base elevation (m, arbitrary datum)", value=0.0, key="db_1d")
            dam_crest = dam_base + dam_height
            reservoir_level = st.slider("Reservoir level at failure (m)", dam_base, dam_crest,
                                         dam_base + 0.9 * dam_height, key="rl_1d")
            failure_mode = st.selectbox("Failure mode", ["piping", "overtopping"], key="fm_1d")

        run_clicked = st.button("Run 1D simulation", type="primary", key="run_1d")

    if not run_clicked:
        st.info("Configure the scenario in the sidebar and click **Run 1D simulation**.")
        return

    with st.spinner("Solving 1D Saint-Venant equations..."):
        if scenario.startswith("Instantaneous"):
            x, bed, result, gauge_idx = _run_instantaneous_1d(
                L_km * 1000, N, dam_km, h_up, h_down, manning_n, width, t_end_min
            )
            breach_info = None
        else:
            x, bed, result, gauge_idx, breach_params, (t_b, Q_b) = _run_breach_1d(
                L_km * 1000, N, dam_km, dam_height, reservoir_vol_mcm, reservoir_level,
                dam_crest, dam_base, failure_mode, h_down, manning_n, width, t_end_min,
            )
            breach_info = breach_params

    if breach_info:
        c1, c2, c3 = st.columns(3)
        c1.metric("Breach avg. width", f"{breach_info['Bavg']:.1f} m")
        c2.metric("Breach formation time", f"{breach_info['tf_hours']:.2f} hr")
        c3.metric("Peak breach outflow", f"{Q_b.max():,.0f} m³/s")

        fig_b = go.Figure()
        fig_b.add_trace(go.Scatter(x=t_b / 60, y=Q_b, name="Breach outflow Q(t)"))
        fig_b.update_layout(title="Breach Outflow Hydrograph (at dam)",
                             xaxis_title="Time (min)", yaxis_title="Discharge (m³/s)",
                             height=300)
        st.plotly_chart(fig_b, use_container_width=True)

    t_arr = result["t"]
    h_arr = result["h"]
    frame_idx = st.slider("Time step", 0, len(t_arr) - 1, len(t_arr) // 2,
                           format="frame %d", key="frame_1d") if len(t_arr) > 1 else 0
    wse = bed + h_arr[frame_idx]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x / 1000, y=bed, name="Bed", line=dict(color="saddlebrown")))
    fig.add_trace(go.Scatter(x=x / 1000, y=wse, name="Water surface", fill="tonexty",
                              line=dict(color="royalblue")))
    for km, idx in gauge_idx.items():
        fig.add_vline(x=km, line_dash="dot", line_color="gray",
                       annotation_text=f"{km:g} km gauge")
    fig.add_vline(x=dam_km, line_color="red", annotation_text="dam/breach")
    fig.update_layout(
        title=f"Water Surface Profile at t = {t_arr[frame_idx] / 60:.1f} min",
        xaxis_title="Chainage (km)", yaxis_title="Elevation (m)", height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Downstream Hydrographs")
    gauges = result["gauges"]
    summary_rows = []
    fig_h = go.Figure()
    fig_q = go.Figure()
    for km, idx in gauge_idx.items():
        g = gauges[idx]
        tt = np.array(g["t"]) / 60
        hh = np.array(g["h"])
        qq = np.array(g["Q"])
        fig_h.add_trace(go.Scatter(x=tt, y=hh, name=f"{km:g} km"))
        fig_q.add_trace(go.Scatter(x=tt, y=qq, name=f"{km:g} km"))

        if qq.max() > 1e-6:
            arrival_idx = int(np.argmax(qq > 0.05 * qq.max()))
            summary_rows.append({
                "Gauge (km)": km,
                "Wave arrival (min)": round(tt[arrival_idx], 1),
                "Peak stage (m)": round(hh.max(), 2),
                "Peak discharge (m³/s)": round(qq.max(), 1),
                "Time to peak (min)": round(tt[int(np.argmax(qq))], 1),
            })

    fig_h.update_layout(title="Stage Hydrographs", xaxis_title="Time (min)",
                         yaxis_title="Depth (m)", height=350)
    fig_q.update_layout(title="Discharge Hydrographs", xaxis_title="Time (min)",
                         yaxis_title="Discharge (m³/s)", height=350)
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_h, use_container_width=True)
    col2.plotly_chart(fig_q, use_container_width=True)

    if summary_rows:
        st.markdown("### Wave Arrival & Peak Summary")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


# =============================================================================
# Tab 2: 2D floodplain + 3D surface visualization
# =============================================================================

def _render_2d_tab():
    st.caption(
        "2D depth-averaged Saint-Venant floodplain simulation (engines.dam_break.SaintVenant2D), "
        "shown as a top-down inundation heatmap and as a 3D surface of the water-surface elevation."
    )
    st.info(
        "**On '3D':** this is a 3D *visualization* of the 2D depth-averaged solve above -- "
        "not an independent 3D (free-surface Navier-Stokes) solver. Saint-Venant is inherently "
        "depth-averaged; a true 3D CFD model is a substantially larger undertaking.",
        icon="ℹ️",
    )

    scenario_2d = st.radio(
        "2D Scenario",
        ["Instantaneous burst (single high-water block)",
         "Progressive breach (Froehlich hydrograph fed in as a source)"],
        horizontal=True, key="scenario_2d",
    )

    with st.sidebar:
        st.header("2D Floodplain Grid")
        domain_w_km = st.slider("Floodplain width, x (km)", 1.0, 20.0, 6.0, key="dw_2d")
        domain_h_km = st.slider("Floodplain length, y (km)", 1.0, 20.0, 4.0, key="dh2_2d")
        dx = st.select_slider("Cell size (m)", [50, 100, 150, 200], value=100, key="dx_2d")
        nx = max(10, int(domain_w_km * 1000 / dx))
        ny = max(10, int(domain_h_km * 1000 / dx))
        st.caption(f"Grid: {nx} × {ny} cells ({nx*ny:,} total)")
        manning_n_2d = st.slider("Manning's n (floodplain)", 0.030, 0.100, 0.050, key="n_2d")
        floodplain_depth = st.slider("Ambient floodplain depth (m)", 0.0, 2.0, 0.1, key="fp_2d")
        breach_width_m = st.slider("Breach opening width (m)", 20, 500, 100, key="bw_2d")
        t_end_min_2d = st.slider("Simulation duration (min)", 15, 180, 60, key="tend_2d")

        st.header("2D Source")
        if scenario_2d.startswith("Instantaneous"):
            reservoir_depth_2d = st.slider("Release-block water depth (m)", 1.0, 15.0, 6.0, key="rd_2d")
            reservoir_extent_km = st.slider("Release-block extent into floodplain (km)", 0.1, 2.0, 0.5, key="re_2d")
        else:
            dam_height_2d = st.slider("Dam height (m)", 5.0, 60.0, 20.0, key="dh2d")
            reservoir_vol_2d = st.slider("Reservoir storage at failure (Mm^3)", 1.0, 500.0, 50.0, key="rv2d")
            dam_base_2d = st.number_input("Dam base elevation (m, datum)", value=0.0, key="db2d")
            dam_crest_2d = dam_base_2d + dam_height_2d
            reservoir_level_2d = st.slider("Reservoir level at failure (m)", dam_base_2d, dam_crest_2d,
                                            dam_base_2d + 0.9 * dam_height_2d, key="rl2d")
            failure_mode_2d = st.selectbox("Failure mode", ["piping", "overtopping"], key="fm2d")

        run_2d = st.button("Run 2D simulation", type="primary", key="run_2d")

    if not run_2d:
        st.info("Configure the 2D floodplain scenario in the sidebar and click **Run 2D simulation**.")
        return

    record_every_s = max(t_end_min_2d * 60 / 30, 15)

    with st.spinner("Solving 2D Saint-Venant equations (dimensional splitting)..."):
        if scenario_2d.startswith("Instantaneous"):
            result, (y0, y1) = _run_2d_instantaneous(
                nx, ny, dx, reservoir_depth_2d, floodplain_depth,
                reservoir_extent_km, breach_width_m, manning_n_2d,
                t_end_min_2d, record_every_s,
            )
            breach_info_2d, Q_b_2d, t_b_2d = None, None, None
        else:
            result, (y0, y1), breach_info_2d, (t_b_2d, Q_b_2d) = _run_2d_progressive_breach(
                nx, ny, dx, dam_height_2d, reservoir_vol_2d, reservoir_level_2d,
                dam_base_2d, failure_mode_2d, breach_width_m, floodplain_depth,
                manning_n_2d, t_end_min_2d, record_every_s,
            )

    if breach_info_2d:
        c1, c2, c3 = st.columns(3)
        c1.metric("Breach avg. width", f"{breach_info_2d['Bavg']:.1f} m")
        c2.metric("Breach formation time", f"{breach_info_2d['tf_hours']:.2f} hr")
        c3.metric("Peak breach outflow", f"{Q_b_2d.max():,.0f} m³/s")

        fig_b2 = go.Figure()
        fig_b2.add_trace(go.Scatter(x=t_b_2d / 60, y=Q_b_2d, name="Breach outflow Q(t)"))
        fig_b2.update_layout(title="Breach Outflow Hydrograph (fed into 2D domain)",
                              xaxis_title="Time (min)", yaxis_title="Discharge (m³/s)", height=280)
        st.plotly_chart(fig_b2, use_container_width=True)

    t_arr2 = result["t"]
    h_frames = result["h"]  # (n_records, ny, nx)
    x_km = np.arange(nx) * dx / 1000
    y_km = np.arange(ny) * dx / 1000

    frame_idx2 = st.slider("Time step", 0, len(t_arr2) - 1, len(t_arr2) - 1,
                            format="frame %d", key="frame_2d") if len(t_arr2) > 1 else 0
    depth_grid = h_frames[frame_idx2]

    col_heat, col_surf = st.columns(2)

    with col_heat:
        fig_heat = go.Figure(data=go.Heatmap(
            z=depth_grid, x=x_km, y=y_km, colorscale="Blues",
            colorbar=dict(title="Depth (m)"),
        ))
        fig_heat.add_shape(type="line", x0=0, x1=0, y0=y_km[y0], y1=y_km[min(y1, ny - 1)],
                            line=dict(color="red", width=6), name="breach")
        fig_heat.update_layout(
            title=f"2D Inundation Depth -- t = {t_arr2[frame_idx2] / 60:.1f} min",
            xaxis_title="x (km)", yaxis_title="y (km)", height=450,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_surf:
        fig_surf = go.Figure(data=go.Surface(
            z=depth_grid, x=x_km, y=y_km, colorscale="Blues",
            colorbar=dict(title="WSE (m)"),
        ))
        fig_surf.update_layout(
            title=f"3D Water-Surface Visualization -- t = {t_arr2[frame_idx2] / 60:.1f} min",
            scene=dict(xaxis_title="x (km)", yaxis_title="y (km)", zaxis_title="Depth (m)",
                       aspectratio=dict(x=1.4, y=1, z=0.4)),
            height=450, margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_surf, use_container_width=True)

    if inundation_extent_stats is not None:
        stats = inundation_extent_stats(depth_grid, cell_area_m2=dx * dx, depth_threshold_m=0.1)
        c1, c2, c3 = st.columns(3)
        c1.metric("Inundated area", f"{stats['inundated_area_km2']:.2f} km²")
        c2.metric("Max depth", f"{stats['max_depth_m']:.2f} m")
        c3.metric("Mean wetted depth", f"{stats['mean_wetted_depth_m']:.2f} m")

    st.markdown("### Depth at a Floodplain Point Over Time")
    px_km = st.slider("Point x (km)", 0.0, float(x_km[-1]), float(x_km[-1] * 0.4), key="px_2d")
    py_km = st.slider("Point y (km)", 0.0, float(y_km[-1]), float(y_km[-1] * 0.5), key="py_2d")
    ix = int(np.clip(round(px_km * 1000 / dx), 0, nx - 1))
    iy = int(np.clip(round(py_km * 1000 / dx), 0, ny - 1))
    point_series = h_frames[:, iy, ix]

    fig_pt = go.Figure()
    fig_pt.add_trace(go.Scatter(x=t_arr2 / 60, y=point_series, mode="lines+markers"))
    fig_pt.update_layout(title=f"Depth at ({px_km:.1f} km, {py_km:.1f} km)",
                          xaxis_title="Time (min)", yaxis_title="Depth (m)", height=300)
    st.plotly_chart(fig_pt, use_container_width=True)


# =============================================================================
# Entry point
# =============================================================================

def render_dam_break_tab():
    st.subheader("Dam-Break / Embankment Breach Flood-Wave Simulator")

    tab_1d, tab_2d = st.tabs(["1D Channel Routing", "2D / 3D Floodplain Inundation"])
    with tab_1d:
        _render_1d_tab()
    with tab_2d:
        _render_2d_tab()


if __name__ == "__main__":
    st.set_page_config(page_title="Dam-Break Lab", layout="wide")
    render_dam_break_tab()