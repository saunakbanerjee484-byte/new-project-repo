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

class AdvancedFilterCriteriaEngine:
    # Initializes the computational brain to isolate geotechnical logic from the presentation layer.
    def __init__(self):
        # Sets the default maximum empirical retention ratio for non-dispersive soils based on Terzaghi.
        self.standard_terzaghi_limit = 5.0
        # Sets the absolute maximum allowable D15f size in millimeters for highly erosive dispersive clays.
        self.dispersive_d15f_limit = 0.5
        # Establishes the upper limit for the coefficient of uniformity to prevent broad grading and segregation.
        self.max_uniformity_coefficient = 20.0

    # ==========================================
    # MODULE 1 STARTS HERE
    # ==========================================
    # Defines the function to evaluate piping resistance and internal stability against segregation.
    def evaluate_retention_and_stability(self, d15f, d85b, d60f, d10f, is_dispersive):
        # Initializes a dictionary to store the calculated geotechnical state and boolean safety flags.
        results = {}
        
        # Calculates the fundamental retention ratio determining if base particles can pass through filter voids.
        retention_ratio = d15f / d85b
        # Stores the calculated retention ratio in the results dictionary for later UI consumption.
        results['retention_ratio'] = retention_ratio
        
        # Checks the boolean flag to determine if the base soil contains easily deflocculated dispersive clay particles.
        if is_dispersive:
            # Overrides standard rules because dispersive soils wash out molecule by molecule rather than by grain bridging.
            results['retention_safe'] = d15f <= self.dispersive_d15f_limit
            # Sets the active maximum threshold to the strict dispersive limit for frontend gauge chart dynamic scaling.
            results['active_retention_limit'] = self.dispersive_d15f_limit
        # Executes the standard empirical check if the base soil is not chemically dispersive.
        else:
            # Validates that the filter voids are small enough to trap the coarser 85 percent fraction of the base soil.
            results['retention_safe'] = retention_ratio <= self.standard_terzaghi_limit
            # Retains the traditional threshold value of 5.0 for standard sand and gravel filter designs.
            results['active_retention_limit'] = self.standard_terzaghi_limit
            
        # Calculates the coefficient of uniformity to quantify the spread of the filter's particle size distribution curve.
        cu = d60f / d10f
        # Stores the uniformity coefficient to be sent to the frontend delta metrics.
        results['uniformity_coefficient'] = cu
        
        # Evaluates if the filter is too broadly graded, which could cause larger particles to roll away from fines during placement.
        results['segregation_safe'] = cu <= self.max_uniformity_coefficient
        
        # Returns the packaged dictionary containing all internal stability and retention assessments.
        return results

    # ==========================================
    # MODULE 2 STARTS HERE
    # ==========================================
    # Defines the function for modern Constriction Size Distribution analysis based on probabilistic void spaces.
    def calculate_csd_retention(self, d_c35, d85b):
        # Initializes a distinct dictionary for the advanced mathematical void-space modeling outputs.
        csd_results = {}
        
        # Stores the controlling constriction size, representing the critical void diameter restricting base soil movement.
        csd_results['controlling_constriction_size'] = d_c35
        
        # Compares the mathematical void diameter directly against the 85th percentile base soil particle diameter.
        csd_results['csd_retention_safe'] = d85b > d_c35
        
        # Calculates the margin of safety by subtracting the constriction size from the base soil size.
        csd_results['csd_safety_margin'] = d85b - d_c35
        
        # Returns the computed CSD parameters to override empirical rules in the advanced UI tab.
        return csd_results

    # ==========================================
    # MODULE 3 STARTS HERE
    # ==========================================
    # Defines the function to ensure the filter can adequately discharge seepage without building hazardous pore pressures.
    def calculate_hydraulic_thickness(self, seepage_discharge_q, filter_permeability_k, hydraulic_gradient_i):
        # Initializes the dictionary for Darcy's Law flow capacity calculations.
        hydraulic_results = {}
        
        # Applies Darcy's Law rearranged to solve for the absolute minimum cross-sectional flow area required.
        required_flow_area = seepage_discharge_q / (filter_permeability_k * hydraulic_gradient_i)
        # Stores the required area to dictate the physical geometry of the constructed filter zone.
        hydraulic_results['required_minimum_area_sqm'] = required_flow_area
        
        # Returns the dimensional requirements to activate the downstream sequential calculation UI cards.
        return hydraulic_results

    # ==========================================
    # MODULE 4 STARTS HERE (UI Payload Generation)
    # ==========================================
    # Defines the function to aggregate all engine calculations into a clean payload for the visual presentation layer.
    def generate_ui_state_payload(self, d15f, d85b, d60f, d10f, is_dispersive, provided_area, required_area):
        # Instantiates the master dictionary that will act as the direct data source for the Streamlit face.
        ui_payload = {}
        
        # Calls the Module 1 method to process empirical limits and internal segregation risks.
        mod1 = self.evaluate_retention_and_stability(d15f, d85b, d60f, d10f, is_dispersive)
        
        # Extracts the raw ratio to position the needle on the frontend Plotly gauge chart.
        ui_payload['gauge_value'] = mod1['retention_ratio']
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

    # ==========================================
    # MODULE 5: Dashboard Bridge API
    # ==========================================
    # Provides the unified evaluate_base_criteria interface expected by geotechnical_dashboard.py,
    # bridging the dashboard's parameter/key conventions to the engine's internal methods.
    def evaluate_base_criteria(self, d15f, d85b, d15b, d50f, d50b, d60f, d10f, is_dispersive):
        # Delegates retention and segregation checks to the existing Module 1 engine method.
        mod1 = self.evaluate_retention_and_stability(d15f, d85b, d60f, d10f, is_dispersive)

        # Builds the result dictionary using the key names the dashboard UI expects.
        result = {}

        # Piping/retention ratio (D15f / D85b) -- already computed inside mod1 as retention_ratio.
        result['piping_ratio'] = mod1['retention_ratio']
        result['retention_safe'] = mod1['retention_safe']
        # Dynamic threshold used by the gauge chart green/red boundary and badge text.
        result['active_retention_limit'] = mod1['active_retention_limit']

        # Permeability ratio (D15f / D15b) -- needs >= 5 for adequate drainage.
        permeability_ratio = d15f / d15b if d15b > 0 else float('inf')
        result['permeability_ratio'] = permeability_ratio
        result['permeability_safe'] = permeability_ratio >= 5.0

        # Gradation ratio (D50f / D50b) -- optional, only when both values are provided.
        if d50f is not None and d50b is not None and d50b > 0:
            gradation_ratio = d50f / d50b
            result['gradation_ratio'] = gradation_ratio
            result['gradation_safe'] = gradation_ratio <= 25.0
        else:
            result['gradation_ratio'] = 0.0
            result['gradation_safe'] = True

        # Segregation (uniformity coefficient Cu) -- forwarded directly from Module 1.
        result['uniformity_coefficient'] = mod1['uniformity_coefficient']
        result['segregation_safe'] = mod1['segregation_safe']

        return result


# Alias so the dashboard can import FilterCriteriaEngine by its expected name.
FilterCriteriaEngine = AdvancedFilterCriteriaEngine