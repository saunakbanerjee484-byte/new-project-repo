readme_content = """# 🌊 DeltaPulse Command Center

**Real-Time Flood Intelligence & Geotechnical Workstation**

Welcome to DeltaPulse, an advanced, production-grade hydro-meteorological intelligence and geotechnical engineering workstation. Built for rigorous academic and engineering standards, this platform integrates real-time basin telemetry, advanced hydraulic routing engines, and a specialized geotechnical workstation for earth dam safety analysis.

---

## 🚀 Core Features & Implemented Modules

### 📍 1. Live Basin Status & Telemetry
* **Real-Time Monitoring:** River water levels cross-evaluated against Central Water Commission (CWC) district thresholds.
* **Dynamic Speedometers:** Interactive Plotly gauge charts displaying dynamic basin health and warning zones.
* **Basin-Wise Inventory:** Expandable categories with native progress indicators representing basin capacity status.

### 📈 2. Advanced Master Hydrographs Workstation
* **Multi-Station Overlay:** Compare stage hydrographs simultaneously across key barrages and observation points.
* **St. Venant Wave Celerity:** Evaluate flood wave propagation speeds dynamically.
* **Hydrological Routing & Separation:** Built-in automated baseflow separation and Muskingum routing optimization (K and X parameters).
* **Synthetic Storm Generator:** Simulate extreme events like monsoon fronts or heavy cloudburst scenarios with dual-axis hyetographs.

### ⛰️ 3. Geotechnical Workstation (Earth Dam Seepage Suite)
* **Specialized UI:** Features a warm amber and terracotta glassmorphism theme representing the intersection of soil mechanics and fluid dynamics.
* **Casagrande Base Parabola & Phreatic Line:** Computes steady-state unconfined seepage profiles with 0.3L upstream corrections and empirical exit-face adjustments.
* **Anisotropic Permeability Analysis:** Evaluates stratified dam cores using transformed-section methods ($k_x \\neq k_z$).
* **Soil Type Auto-Selector:** Instantaneous order-of-magnitude property lookups for gravels, sands, and silty clays.
* **Terzaghi Filter Criteria Checker:** Validates granular stability ratios ($D_{15f}/D_{85b}$ and $D_{15f}/D_{15b}$) to prevent piping failure.
* **Seed-Idriss Liquefaction Module:** Simplified evaluation of cyclic stress ratios (CSR) versus SPT-corrected cyclic resistance ratios (CRR).

---

## 🗺️ Upcoming Geotechnical Roadmap (Planned Features)

The following advanced modules are systematically structured in the platform roadmap. They are slated for future releases utilizing dedicated numerical solvers and verified geotechnical frameworks:

* 🔜 **Bishop's Simplified & Janbu's Generalized Slope Stability:** Circular and non-circular slip-surface grid searches paired with rigorous slice-by-slice normal and shear force equilibrium.
* 🔜 **Rapid Drawdown Transient Seepage:** Pore-pressure ratio modeling ($r_u$) via Bishop/Morgenstern procedures during critical reservoir level drops.
* 🔜 **Progressive Piping & Internal Erosion Rate:** Hanson erodibility-index methodology coupled with pipe-widening temporal tracking.
* 🔜 **Frost Heave & Freeze-Thaw Pore Pressure:** Konrad & Morgenstern segregation-potential (SP) modeling for cold-region embankment behavior.
* 🔜 **Geosynthetic Reinforcement Pullout:** Federal Highway Administration (FHWA) pullout capacity calculations incorporating manufacturer-specific interaction coefficients.
* 🔜 **Wave Overtopping & Crest Erosion Rate:** Broad-crested weir outflow mechanics paired with surface erodibility tracking.
* 🔜 **Saturated-Unsaturated Transient Seepage:** Finite-element/finite-difference time-stepping solutions of Richards' equation utilizing van Genuchten soil-water characteristic curves.

---

## 💻 How to Run the Application

Follow these steps to set up and launch the DeltaPulse Command Center locally on your machine.

### Prerequisites
* Python 3.9+ is required.
* Ensure you are operating within the root directory (`wb_flood_intelligence/`).

### Step 1: Install Dependencies
Install the required scientific and visualization libraries via pip:
```bash
pip install streamlit pandas numpy plotly
### Step 2 :Step 2: Clear Python Bytecode Cache (Recommended)
To ensure clean imports and prevent the system from using stale bytecode after structural updates, run these commands in your terminal.
### For Windows:
python -Bc "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.py[co]')]"
python -Bc "import pathlib; [p.rmdir() for p in pathlib.Path('.').rglob('__pycache__') if p.is_dir()]"
###Step 3: Launch the Command Center
Start the local Streamlit web server:
python -m streamlit run app.py