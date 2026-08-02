"""
app.py

DeltaPulse Command Center -- Streamlit dashboard.
Strictly mapped to the project V-max.zip architecture.
"""

import streamlit as st
import pandas as pd
import json
import os

# Configure page settings
st.set_page_config(page_title="DeltaPulse - WB Flood Intelligence", layout="wide", page_icon="🌊")

def render_live_status():
    st.header("📍 Live Status: District Thresholds")
    st.markdown("Real-time gauge levels vs. district thresholds across West Bengal.")
    
    thresholds_path = os.path.join("config", "thresholds.json")
    
    try:
        with open(thresholds_path, "r") as f:
            thresholds_data = json.load(f)
            
        districts = thresholds_data.get("districts", {})
        
        st.subheader("Critical Warning & Danger Levels")
        
        # Display top 3 stations as highlight metric cards
        st.markdown("##### 🔍 Highlighted Stations")
        cols = st.columns(3)
        highlight_districts = list(districts.items())[:3]
        
        for i, (district_name, data) in enumerate(highlight_districts):
            clean_name = district_name.replace("_", " ")
            with cols[i]:
                st.metric(
                    label=f"{clean_name} | {data['river']}",
                    value=f"{data['station']}",
                    delta=f"Danger: {data['danger_level_m']} m | Warning: {data['warning_level_m']} m",
                    delta_color="off"
                )
                
        st.divider()
        
        # Convert JSON to a clean Pandas DataFrame for tabular display
        df_list = []
        for district, data in districts.items():
            df_list.append({
                "District": district.replace("_", " "),
                "River": data["river"],
                "Station": data["station"],
                "Warning Level (m)": data["warning_level_m"],
                "Danger Level (m)": data["danger_level_m"]
            })
            
        df = pd.DataFrame(df_list)
        
        st.markdown("##### 📊 Complete District Threshold Data")
        st.dataframe(df, use_container_width=True, hide_index=True)

    except FileNotFoundError:
        st.error(f"Could not find `{thresholds_path}`. Make sure your JSON file is in the config folder.")
    except json.JSONDecodeError:
        st.error("Error reading the JSON file. Please check for syntax issues.")

def render_hydrographs():
    st.header("📈 Hydrographs & Flow Trends")
    st.markdown("Plot Plotly hydrographs per `utils.registry.STATIONS` here.")
    
    try:
        import plotly.express as px
        import numpy as np
        
        dates = pd.date_range(start="2026-08-01", periods=48, freq="h")
        mock_levels = np.sin(np.linspace(0, 10, 48)) * 2 + 15  
        df = pd.DataFrame({"Time": dates, "Water Level (m)": mock_levels})
        
        fig = px.line(df, x="Time", y="Water Level (m)", title="Simulated Station Hydrograph")
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.warning("Hydrographs disabled. Run: `pip install plotly pandas` to activate.")

def render_dam_break():
    # Safely import from the verified calculator module
    try:
        from calculator.dam_break_dashboard import render_dam_break_tab
        render_dam_break_tab()
    except Exception as e:
        st.error(f"Failed to load Dam-Break Lab: {e}")

def render_forecasts():
    st.header("🤖 Machine Learning Predictive Forecaster")
    st.info("Wire this tab to `models.forecaster.WaterLevelForecaster` once `models.trainer` has produced trained model artifacts in `models_store/`.")
    
    # Try importing from the verified models directory
    try:
        from models.forecaster import forecaster
        st.success("Forecaster engine loaded successfully!")
    except Exception as e:
        st.error(f"Forecaster module pending integration: {e}")

def render_embankment():
    st.header("⛰️ Earth Dam Seepage Analysis")
    st.info("Loading physics engine from `engines.embankment`.")
    
    try:
        from engines.embankment import embankment_engine
        st.success("Geotechnical engine loaded successfully!")
    except Exception as e:
        st.error(f"Embankment engine pending integration: {e}")


def main():
    st.title("🌊 DeltaPulse -- West Bengal Flood Intelligence Command Center")
    st.caption("Real-time river analytics · hydraulic & numerical engines · ML forecasting")
    st.divider()

    st.sidebar.title("Navigation")
    st.sidebar.markdown("Select a module to deploy:")
    
    # Navigation maps to specific isolated functions
    pages = {
        "📍 Live Status (Thresholds)": render_live_status,
        "📈 Hydrographs": render_hydrographs,
        "🌊 Dam-Break Lab (1D/2D)": render_dam_break,
        "⛰️ Embankment Geotech": render_embankment,
        "🤖 ML Forecasts": render_forecasts
    }
    
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    
    # Execute the selected page
    pages[selection]()

if __name__ == "__main__":
    main()