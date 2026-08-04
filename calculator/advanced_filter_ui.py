"""
This computational engine serves as the geotechnical backend for designing and validating earth dam embankment filters.
It systematically transitions from empirical retention guidelines to advanced mathematical void-space modeling.
The engine begins by evaluating fundamental Terzaghi criteria while applying strict overrides for vulnerable dispersive base soils.
Internal stability is rigorously assessed using the coefficient of uniformity to prevent hazardous segregation during construction.
Moving beyond empirical rules, the code incorporates Constriction Size Distribution logic to mathematically model filter void spaces.
This advanced approach determines the controlling constriction size to guarantee precise retention of base soil particles.
Once retention and stability are theoretically validated, the engine sequentially shifts focus to hydraulic drainage capacity.
By applying Darcy's Law, it computes the absolute minimum physical thickness required to safely dissipate expected seepage flows.
Finally, the engine structures all calculated metrics, boolean safety flags, and margins into standardized data payloads.
These structured outputs form the essential data layer that will directly drive the dynamic frontend UI and gauge charts.
"""

# Defines the core engine class that isolates pure geotechnical mathematics from the UI presentation layer.
class FilterCriteriaEngine:
    # Initializes the class and establishes the fundamental geotechnical constants and empirical thresholds.
    def __init__(self):
        # Defines Terzaghi's upper limit for the retention ratio to prevent base soil particle migration (piping).
        self.max_retention_ratio = 5.0
        # Defines Terzaghi's lower limit for the permeability ratio to ensure rapid dissipation of seepage water.
        self.min_permeability_ratio = 5.0
        # Defines the upper limit for the gradation ratio to ensure filter and base soil grading curves remain parallel.
        self.max_gradation_ratio = 25.0
        # Imposes a strict maximum D15 size of 0.5 mm when the base soil is identified as a highly erodible dispersive clay.
        self.dispersive_d15f_limit = 0.5
        # Sets the maximum allowable Coefficient of Uniformity (Cu) to 20 to prevent broad grading and particle segregation.
        self.max_uniformity_coefficient = 20.0

    # ==========================================
    # MODULE 1: RETENTION, PERMEABILITY & STABILITY
    # ==========================================
    # Evaluates empirical piping resistance, drainage capacity, and internal filter stability.
    def evaluate_base_criteria(self, d15f, d85b, d15b, d50f, d50b, d60f, d10f, is_dispersive):
        # Initializes a dictionary to encapsulate all resulting ratios and boolean safety validations.
        results = {}
        
        # Calculates the piping ratio (D15f/D85b) to check if filter voids can retain the coarser 85% of base soil.
        results['piping_ratio'] = d15f / d85b if d85b else 0
        # Calculates the permeability ratio (D15f/D15b) to verify the filter is significantly more pervious than the base.
        results['permeability_ratio'] = d15f / d15b if d15b else 0
        
        # Validates if the calculated permeability ratio meets or exceeds the minimum requirement for safe drainage.
        results['permeability_safe'] = results['permeability_ratio'] >= self.min_permeability_ratio
        
        # Conditionally calculates the gradation ratio if D50 values are provided by the user.
        if d50f and d50b:
            # Calculates the D50 ratio to ensure overall similarity in the shape of both particle size distribution curves.
            results['gradation_ratio'] = d50f / d50b
            # Validates that the gradation ratio is within the safe empirical boundary to prevent gap-grading bridging.
            results['gradation_safe'] = results['gradation_ratio'] <= self.max_gradation_ratio
        
        # Triggers a specialized logic override if the base soil is chemically prone to dispersive deflocculation.
        if is_dispersive:
            # Suspends the standard ratio check and directly restricts the filter's D15 to physically block clay colloids.
            results['retention_safe'] = d15f <= self.dispersive_d15f_limit
            # Records the active critical threshold (0.5 mm) to dynamically scale the UI gauge charts.
            results['active_retention_limit'] = self.dispersive_d15f_limit
        # Executes the standard Terzaghi empirical retention check for non-dispersive, standard soil matrices.
        else:
            # Validates that the piping ratio does not exceed the standard limit of 5.0.
            results['retention_safe'] = results['piping_ratio'] <= self.max_retention_ratio
            # Records the standard empirical limit (5.0) for UI gauge chart scaling.
            results['active_retention_limit'] = self.max_retention_ratio
            
        # Calculates the Coefficient of Uniformity (Cu = D60/D10) to quantify the internal grading spread of the filter.
        cu = d60f / d10f if d10f else 0
        # Stores the calculated Cu value to evaluate internal instability and self-healing potential.
        results['uniformity_coefficient'] = cu
        # Determines if the filter grading is uniform enough to resist segregation during transportation and compaction.
        results['segregation_safe'] = cu <= self.max_uniformity_coefficient
        
        # Concludes the module by returning the comprehensive dictionary of empirical and stability diagnostics.
        return results

    # ==========================================
    # MODULE 2: CONSTRICTION SIZE DISTRIBUTION (CSD)
    # ==========================================
    # Assesses filter retention based on mathematical probability models of void spaces rather than solid particles.
    def calculate_csd_analysis(self, d_c35, d85b):
        # Initializes a dictionary to store outputs from the void-space mathematical model.
        csd_results = {}
        # Stores the calculated controlling constriction size (Dc35) representing the critical filter void diameter.
        csd_results['controlling_void_size'] = d_c35
        # Verifies retention by confirming the 85th percentile base particle is larger than the critical filter void.
        csd_results['csd_retention_safe'] = d85b > d_c35
        # Calculates the numerical safety margin representing how much larger the base particle is compared to the void.
        csd_results['csd_safety_margin'] = d85b - d_c35
        # Returns the CSD payload to drive advanced probabilistic heatmaps in the UI layer.
        return csd_results

    # ==========================================
    # MODULE 3: HYDRAULIC CONDUCTIVITY & THICKNESS
    # ==========================================
    # Applies groundwater flow mechanics to determine the physical dimensions required for safe filter drainage.
    def calculate_hydraulic_capacity(self, q_seepage, k_filter, i_gradient):
        # Initializes a dictionary to store physical design parameters derived from fluid mechanics.
        hydro_results = {}
        # Utilizes Darcy's Law (A = q / (k*i)) to mathematically compute the minimum cross-sectional area for flow.
        required_area = q_seepage / (k_filter * i_gradient) if (k_filter and i_gradient) else 0
        # Stores this required dimensional area to act as a baseline for the engineer's physical design thickness.
        hydro_results['required_flow_area'] = required_area
        # Returns the hydraulic payload to be used sequentially after retention criteria are satisfied.
        return hydro_results

    # ==========================================
    # MODULE 4: UI PAYLOAD GENERATION (The Connector)
    # ==========================================
    # Defines the function to aggregate all engine calculations into a clean payload for the visual presentation layer.
    def generate_ui_state_payload(self, d15f, d85b, d15b, d50f, d50b, d60f, d10f, is_dispersive, provided_area, required_area):
        # Instantiates the master dictionary that will act as the direct data source for the Streamlit face.
        ui_payload = {}
        
        # Calls the Module 1 method to process empirical limits and internal segregation risks.
        mod1 = self.evaluate_base_criteria(d15f, d85b, d15b, d50f, d50b, d60f, d10f, is_dispersive)
        
        # Extracts the raw ratio to position the needle on the frontend Plotly gauge chart.
        ui_payload['gauge_value'] = mod1['piping_ratio']
        # Extracts the dynamic limit to define where the green zone ends and the red zone begins on the gauge.
        ui_payload['gauge_limit'] = mod1['active_retention_limit']
        # Triggers a red visual alert and automatically expands the remediation panel if the ratio fails.
        ui_payload['show_retention_alert'] = not mod1['retention_safe']
        
        # Extracts the calculated uniformity coefficient for the dynamic metric text display.
        ui_payload['metric_cu_value'] = mod1['uniformity_coefficient']
        # Calculates the exact numerical difference between the safe boundary and the actual grading spread.
        ui_payload['metric_cu_delta'] = self.max_uniformity_coefficient - mod1['uniformity_coefficient']
        # Opens the Kenney and Lau internal instability expander if the grading curve is too broad.
        ui_payload['show_segregation_expander'] = not mod1['segregation_safe']
        
        # Calculates the surplus drainage capacity to populate the secondary delta indicator in the thickness card.
        ui_payload['hydraulic_margin_delta'] = provided_area - required_area
        
        # Returns the final serialized state that the Streamlit frontend will use to render the Command Center.
        return ui_payload