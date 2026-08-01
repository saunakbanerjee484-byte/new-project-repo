"""
dam_break_dashboard.py

Interactive Streamlit panel for engines.dam_break -- lets a user configure
either a classic instantaneous dam-break or an embankment/earth-dam
breach (Froehlich empirical breach -> broad-crested weir outflow ->
routed downstream via the 1D Saint-Venant HLL solver), then inspect the
resulting water-surface profile animation and downstream hydrographs.

Run standalone:   streamlit run dam_break_dashboard.py
Or import render_dam_break_tab() into app.py as a tab (see app.py).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engines.dam_break import (
    SaintVenant1D, froehlich_breach_params, breach_outflow_hydrograph,
    breach_inflow_function, G,
)


@st.cache_data(show_spinner=False)
def _run_instantaneous(L, N, dam_km, h_up, h_down, manning_n, width, t_end_min):
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
def _run_breach(L, N, dam_km, dam_height, reservoir_vol_mcm, reservoir_level,
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


def render_dam_break_tab():
    st.subheader("Dam-Break / Embankment Breach Flood-Wave Simulator")
    st.caption("1D Saint-Venant (HLL finite-volume) flood-wave routing -- engines/dam_break.py")

    scenario = st.radio(
        "Scenario", ["Instantaneous dam-break (classic)", "Embankment / earth-dam breach"],
        horizontal=True,
    )

    with st.sidebar:
        st.header("Channel")
        L_km = st.slider("Reach length (km)", 5, 100, 30)
        dam_km = st.slider("Dam/barrage chainage (km)", 0.0, float(L_km), 2.0)
        N = st.select_slider("Grid resolution (cells)", [200, 400, 800, 1200], value=400)
        manning_n = st.slider("Manning's n", 0.020, 0.060, 0.035)
        width = st.slider("Channel width (m)", 20, 500, 120)
        h_down = st.slider("Downstream initial depth (m)", 0.0, 5.0, 1.0)
        t_end_min = st.slider("Simulation duration (min)", 30, 360, 120)

        st.header("Reservoir / Dam")
        if scenario.startswith("Instantaneous"):
            h_up = st.slider("Upstream reservoir depth (m)", 2.0, 25.0, 12.0)
        else:
            dam_height = st.slider("Dam height (m)", 5.0, 60.0, 20.0)
            reservoir_vol_mcm = st.slider("Reservoir storage at failure (Mm^3)", 1.0, 500.0, 50.0)
            dam_base = st.number_input("Dam base elevation (m, arbitrary datum)", value=0.0)
            dam_crest = dam_base + dam_height
            reservoir_level = st.slider("Reservoir level at failure (m)", dam_base, dam_crest,
                                         dam_base + 0.9 * dam_height)
            failure_mode = st.selectbox("Failure mode", ["piping", "overtopping"])

        run_clicked = st.button("Run simulation", type="primary")

    if not run_clicked:
        st.info("Configure the scenario in the sidebar and click **Run simulation**.")
        return

    with st.spinner("Solving Saint-Venant equations..."):
        if scenario.startswith("Instantaneous"):
            x, bed, result, gauge_idx = _run_instantaneous(
                L_km * 1000, N, dam_km, h_up, h_down, manning_n, width, t_end_min
            )
            breach_info = None
        else:
            x, bed, result, gauge_idx, breach_params, (t_b, Q_b) = _run_breach(
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

    # --- Water surface profile animation -----------------------------------
    t_arr = result["t"]
    h_arr = result["h"]
    frame_idx = st.slider("Time step", 0, len(t_arr) - 1, len(t_arr) // 2,
                           format="frame %d") if len(t_arr) > 1 else 0
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

    # --- Downstream hydrographs ---------------------------------------------
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


if __name__ == "__main__":
    st.set_page_config(page_title="Dam-Break Lab", layout="wide")
    render_dam_break_tab()
