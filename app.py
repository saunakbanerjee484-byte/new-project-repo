"""
app.py

DeltaPulse Command Center -- Streamlit dashboard.

Tabs:
  - Live Status     : current gauge levels vs. district thresholds (Folium map)
  - Hydrographs      : Plotly time-series for selected stations
  - Dam-Break Lab    : interactive 1D Saint-Venant dam-break / embankment
                        breach simulator (engines.dam_break)
  - Forecasts        : 6h/12h ML water-level predictions

Run:  streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="DeltaPulse - WB Flood Intelligence", layout="wide")

st.title("🌊 DeltaPulse -- West Bengal Flood Intelligence Command Center")
st.caption("Real-time river analytics · hydraulic & numerical engines · ML forecasting")

tab_status, tab_hydro, tab_damBreak, tab_forecast = st.tabs(
    ["Live Status", "Hydrographs", "Dam-Break Lab", "Forecasts"]
)

with tab_status:
    st.info(
        "Wire this tab to workers.scheduler.poll_once() output "
        "(data/processed/alerts_*.json) once live telemetry is flowing, "
        "and render station markers with places_map_display-style Folium map."
    )

with tab_hydro:
    st.info("Plot Plotly hydrographs per utils.registry.STATIONS station here.")

with tab_damBreak:
    # dam_break_dashboard.py now lives in calculator/ (moved from the
    # project root) -- import it as a package module instead.
    from calculator.dam_break_dashboard import render_dam_break_tab
    render_dam_break_tab()

with tab_forecast:
    st.info(
        "Wire this tab to models.forecaster.WaterLevelForecaster once "
        "models.trainer has produced trained model artifacts in models_store/."
    )