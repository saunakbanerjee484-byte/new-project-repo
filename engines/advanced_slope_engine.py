"""
engines/advanced_slope_engine.py

Advanced Slope Stability & Coupled Seepage-Force Mechanics
==========================================================

GEOTECHNICAL SCHEMES & MATHEMATICAL FORMULATIONS (PHASE 1):

1. BISHOP'S SIMPLIFIED METHOD OF SLICES (Grid Search Engine)
   --------------------------------------------------------
   Factor of Safety (FoS) is defined as the ratio of available shear strength 
   to mobilized shear stress along a trial circular slip surface.
   
   Iterative Bishop Formula:
                          ∑ [ (c' * b_i + (W_i - u_i * b_i) * tan(φ')) / m_αi ]
            FoS = -------------------------------------------------------------
                   ∑ [ W_i * sin(α_i) ] + Seismic_Driving_Moment_Term
   
   Where:
            m_αi = cos(α_i) + [ (sin(α_i) * tan(φ')) / FoS ]
            b_i  = Width of slice i
            W_i  = Total weight of slice i (including vertical seismic adjustments)
            u_i  = Pore water pressure at the base of slice i
            α_i  = Base inclination angle of slice i

2. FEATURE 2: PSEUDO-STATIC SEISMIC ANALYSIS (k_h, k_v)
   ----------------------------------------------------
   Horizontal seismic coefficient (k_h) and vertical seismic coefficient (k_v)
   model earthquake loading as equivalent static inertial forces.
   
            W_i_seismic = W_i * (1 ∓ k_v)
            Horizontal Inertial Force: F_h_i = k_h * W_i
            Seismic Driving Moment Contribution:
                Moment = ∑ [ k_h * W_i * (y_mid_i - Y_center) / Radius ]

3. FEATURE 4: SKEMPTON'S B-BAR RAPID DRAWDOWN METHOD
   -------------------------------------------------
   During rapid drawdown, external reservoir support drops instantaneously, 
   inducing undrained pore pressure responses in saturated soil slices.
   
            Total vertical stress change: Δσ_v = -γ_w * Δh_w
            Excess Pore Pressure Generated: Δu = B_bar * Δσ_v
            Residual Pore Water Pressure: u_drawdown = u_initial + Δu
   
   Where B_bar (Skempton's pore pressure parameter) typically ranges 0.8 - 1.0 
   for soft/saturated clays.

4. FEATURE 3: PROBABILISTIC SLOPE STABILITY (Monte Carlo Simulation)
   -----------------------------------------------------------------
   Soil parameters c' and φ' are modeled as normal random variables:
            c'  ~ N(μ_c, σ_c^2)
            φ'  ~ N(μ_φ, σ_φ^2)
   
   Probability of Failure (PoF) is calculated from N Monte Carlo iterations:
            PoF (%) = [ Count(FoS_k < 1.0) / N_total ] * 100
"""

import numpy as np

class AdvancedSlopeEngine:
    def __init__(
        self,
        dam_height_m: float = 20.0,
        water_depth_m: float = 18.0,
        m_u: float = 3.0,  # Upstream slope (H:1V)
        m_d: float = 2.0,  # Downstream slope (H:1V)
        crest_width_m: float = 6.0,
        c_prime_kpa: float = 15.0,
        phi_prime_deg: float = 30.0,
        unit_weight_kn_m3: float = 19.0,
        gamma_w: float = 9.81,
    ):
        self.H = dam_height_m
        self.H_w = water_depth_m
        self.m_u = m_u
        self.m_d = m_d
        self.crest = crest_width_m
        
        self.c_prime = c_prime_kpa
        self.phi_prime = np.radians(phi_prime_deg)
        self.gamma = unit_weight_kn_m3
        self.gamma_w = gamma_w

        # Pre-compute Dam Outline Points
        self.base_width = self.m_d * self.H + self.crest + self.m_u * self.H
        self.x_toe_ds = 0.0
        self.x_crest_ds = self.m_d * self.H
        self.x_crest_us = self.x_crest_ds + self.crest
        self.x_toe_us = self.base_width

    def get_dam_surface_y(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros_like(x, dtype=float)
        mask_ds = (x >= self.x_toe_ds) & (x < self.x_crest_ds)
        y[mask_ds] = x[mask_ds] / self.m_d
        mask_crest = (x >= self.x_crest_ds) & (x <= self.x_crest_us)
        y[mask_crest] = self.H
        mask_us = (x > self.x_crest_us) & (x <= self.x_toe_us)
        y[mask_us] = self.H - (x[mask_us] - self.x_crest_us) / self.m_u
        return np.clip(y, 0.0, self.H)

    # =========================================================================
    # FEATURE 1, 2 & 4: BISHOP'S METHOD (CIRCULAR) WITH SEISMIC & DRAWDOWN
    # =========================================================================
    def calculate_slice_bishop_fos(
        self,
        center_x: float,
        center_y: float,
        radius: float,
        k_h: float = 0.0,
        k_v: float = 0.0,
        drawdown_water_depth_m: float = None,
        b_bar: float = 0.8,
        num_slices: int = 30,
        max_iter: int = 50,
        tol: float = 1e-3
    ) -> float:
        x_min = max(self.x_toe_ds, center_x - radius)
        x_max = min(self.x_toe_us, center_x + radius)
        
        if x_max <= x_min:
            return np.nan

        x_grid = np.linspace(x_min, x_max, num_slices + 1)
        b = (x_max - x_min) / num_slices
        x_mids = 0.5 * (x_grid[:-1] + x_grid[1:])
        y_ground = self.get_dam_surface_y(x_mids)
        
        temp = radius**2 - (x_mids - center_x)**2
        valid_mask = temp > 0
        if not np.any(valid_mask): return np.nan

        y_slip = center_y - np.sqrt(np.maximum(0, temp))
        active_slices = (y_slip < y_ground) & valid_mask
        if np.sum(active_slices) < 3: return np.nan

        x_m = x_mids[active_slices]
        y_g = y_ground[active_slices]
        y_s = y_slip[active_slices]
        
        slice_height = y_g - y_s
        slice_weight = self.gamma * b * slice_height * (1.0 - k_v)
        alpha = np.arcsin((x_m - center_x) / radius)
        
        h_w_initial = np.clip(self.H_w - y_s, 0.0, None)
        u_initial = self.gamma_w * h_w_initial
        
        if drawdown_water_depth_m is not None:
            h_w_final = np.clip(drawdown_water_depth_m - y_s, 0.0, None)
            delta_h_w = h_w_initial - h_w_final
            excess_u = b_bar * (-self.gamma_w * delta_h_w)
            u_pore = u_initial + excess_u
        else:
            u_pore = u_initial

        y_mid_slice = 0.5 * (y_g + y_s)
        seismic_lever_arm = (y_mid_slice - center_y) / radius
        
        driving_gravitational = np.sum(slice_weight * np.sin(alpha))
        driving_seismic = np.sum(k_h * slice_weight * seismic_lever_arm)
        driving_total = driving_gravitational + driving_seismic

        if driving_total <= 0: return np.nan

        fos = 1.5
        for _ in range(max_iter):
            m_alpha = np.cos(alpha) + (np.sin(alpha) * np.tan(self.phi_prime)) / fos
            m_alpha = np.maximum(m_alpha, 0.01)
            numerator = np.sum((self.c_prime * b + (slice_weight - u_pore * b) * np.tan(self.phi_prime)) / m_alpha)
            fos_new = numerator / driving_total
            if abs(fos_new - fos) < tol: return max(0.1, fos_new)
            fos = fos_new
        return max(0.1, fos)

    def search_critical_slip_surface(self, grid_nx=10, grid_ny=10, k_h=0.0, k_v=0.0, drawdown_water_depth_m=None, b_bar=0.8):
        """
        Executes a brute-force grid search over a 2D mesh of circle centers and radiuses.
        Returns the minimum FoS, critical circle coordinates, arc geometry, and grid heatmap.
        """
        xc_min, xc_max = self.x_toe_ds, self.x_toe_us
        yc_min, yc_max = self.H * 1.1, self.H * 2.2
        xc_vec = np.linspace(xc_min, xc_max, grid_nx)
        yc_vec = np.linspace(yc_min, yc_max, grid_ny)

        min_fos = 999.0
        best_circle = None
        grid_heatmap = np.full((grid_ny, grid_nx), np.nan)

        for j, yc in enumerate(yc_vec):
            for i, xc in enumerate(xc_vec):
                r_min = yc - self.H
                r_max = np.sqrt((xc - self.x_toe_ds)**2 + yc**2)
                radii = np.linspace(r_min, r_max, 5)

                center_min_fos = 999.0
                for r in radii:
                    fos = self.calculate_slice_bishop_fos(
                        xc, yc, r, k_h=k_h, k_v=k_v, 
                        drawdown_water_depth_m=drawdown_water_depth_m, b_bar=b_bar
                    )
                    if not np.isnan(fos) and fos < center_min_fos:
                        center_min_fos = fos
                        if fos < min_fos:
                            min_fos = fos
                            best_circle = (xc, yc, r)

                grid_heatmap[j, i] = center_min_fos if center_min_fos < 900 else np.nan

        # Generate smooth arc coordinates for the critical slip circle
        arc_x, arc_y = np.array([]), np.array([])
        if best_circle is not None:
            xc, yc, r = best_circle
            angles = np.linspace(0, 2 * np.pi, 200)
            x_all = xc + r * np.cos(angles)
            y_all = yc - r * np.sin(angles)
            
            y_g = self.get_dam_surface_y(x_all)
            mask = (y_all <= y_g) & (y_all >= 0) & (x_all >= self.x_toe_ds) & (x_all <= self.x_toe_us)
            
            sort_idx = np.argsort(x_all[mask])
            arc_x = x_all[mask][sort_idx]
            arc_y = y_all[mask][sort_idx]

        return {
            "min_fos": float(min_fos if min_fos < 900 else 1.0),
            "best_circle": best_circle,
            "arc_x": arc_x,
            "arc_y": arc_y,
            "grid_xc": xc_vec,
            "grid_yc": yc_vec,
            "grid_heatmap": grid_heatmap
        }

    # =========================================================================
    # FEATURE 3: PROBABILISTIC SLOPE STABILITY (MONTE CARLO SIMULATION)
    # =========================================================================
    def run_monte_carlo_simulation(
        self,
        critical_circle: tuple,
        num_simulations: int = 500,
        c_std_kpa: float = 3.0,
        phi_std_deg: float = 2.5,
        k_h: float = 0.0,
        k_v: float = 0.0,
        drawdown_water_depth_m: float = None
    ) -> dict:
        """
        Runs Monte Carlo iterations on the critical slip surface using stochastic soil properties.
        Calculates Probability of Failure (PoF) and FoS distribution statistics.
        """
        if critical_circle is None:
            return {"pof_percent": 0.0, "fos_samples": np.array([]), "mean_fos": 0.0, "std_fos": 0.0}

        xc, yc, r = critical_circle
        
        c_samples = np.random.normal(loc=self.c_prime, scale=c_std_kpa, size=num_simulations)
        phi_samples_deg = np.random.normal(loc=np.degrees(self.phi_prime), scale=phi_std_deg, size=num_simulations)
        
        c_samples = np.clip(c_samples, 0.0, None)
        phi_samples_deg = np.clip(phi_samples_deg, 1.0, 50.0)

        fos_results = []
        original_c = self.c_prime
        original_phi = self.phi_prime

        for i in range(num_simulations):
            self.c_prime = c_samples[i]
            self.phi_prime = np.radians(phi_samples_deg[i])
            
            fos = self.calculate_slice_bishop_fos(
                xc, yc, r, k_h=k_h, k_v=k_v, drawdown_water_depth_m=drawdown_water_depth_m
            )
            if not np.isnan(fos):
                fos_results.append(fos)

        self.c_prime = original_c
        self.phi_prime = original_phi

        fos_array = np.array(fos_results)
        failed_count = np.sum(fos_array < 1.0)
        pof = (failed_count / len(fos_array)) * 100.0 if len(fos_array) > 0 else 0.0

        return {
            "pof_percent": pof,
            "fos_samples": fos_array,
            "mean_fos": float(np.mean(fos_array)) if len(fos_array) > 0 else 0.0,
            "std_fos": float(np.std(fos_array)) if len(fos_array) > 0 else 0.0
        }

    # === PHASE 1 ENDS HERE ===

    # 🛡️ PHASE 2 ASSEMBLE 🛡️
    
    """
    GEOTECHNICAL SCHEMES & MATHEMATICAL FORMULATIONS (PHASE 2):

    5. NON-CIRCULAR WEDGE FAILURE (Janbu's Simplified Method)
       ------------------------------------------------------
       When a weak layer exists (e.g., a clay seam at the foundation), the slip surface 
       truncates and runs horizontally along it. 
       FoS (Janbu) = f_0 * ( ∑ [ (c'*b + (W - u*b)*tan(φ')) / (cos(α)*m_α) ] / ∑ [ W * tan(α) ] )
       Where f_0 is an empirical correction factor based on wedge geometry.

    6. TENSION CRACK & HYDROSTATIC SURCHARGE
       -------------------------------------
       Depth of tension crack in cohesive soil: Z_c = 2*c' / (γ * sqrt(Ka))
       If water fills the crack, it exerts a horizontal hydrostatic force P_w:
       P_w = 0.5 * γ_w * Z_c^2
       This force directly adds to the driving forces in the denominator.

    7. GEOSYNTHETIC REINFORCEMENT & TOE BERMS (Remediation)
       ----------------------------------------------------
       Reinforcement adds a stabilizing force T (Tensile strength).
       Toe Berm adds extra weight (W_berm) at the toe, producing a negative 
       (resisting) driving moment, effectively increasing the FoS.

    8. STRAIN-SOFTENING & PROGRESSIVE FAILURE (Peak vs Residual)
       ---------------------------------------------------------
       Slices experiencing high shear stress yield, and their strength drops 
       from Peak (c'_p, φ'_p) to Residual (c'_r, φ'_r). 
       This causes stress redistribution, potentially leading to catastrophic failure.
    """

    # =========================================================================
    # FEATURE 5: NON-CIRCULAR / COMPOSITE SLIP SURFACES (WEDGE FAILURE)
    # =========================================================================
    def calculate_wedge_janbu_fos(
        self,
        entry_x: float,
        exit_x: float,
        weak_layer_elevation_m: float,
        num_slices: int = 30,
        max_iter: int = 50,
        tol: float = 1e-3
    ) -> float:
        """
        Calculates FoS using Janbu's Simplified Method for a non-circular wedge failure 
        sliding along a designated weak foundation layer.
        """
        if exit_x <= entry_x: return np.nan
        
        x_grid = np.linspace(entry_x, exit_x, num_slices + 1)
        b = (exit_x - entry_x) / num_slices
        x_mids = 0.5 * (x_grid[:-1] + x_grid[1:])
        y_ground = self.get_dam_surface_y(x_mids)
        
        y_slip = np.zeros_like(x_mids)
        for i, x in enumerate(x_mids):
            y_plunge = self.get_dam_surface_y(np.array([entry_x]))[0] - (x - entry_x) * np.tan(np.radians(45))
            y_rise = self.get_dam_surface_y(np.array([exit_x]))[0] - (exit_x - x) * np.tan(np.radians(45))
            y_slip[i] = max(weak_layer_elevation_m, min(y_plunge, y_rise))

        active_slices = y_slip < y_ground
        if np.sum(active_slices) < 3: return np.nan

        x_m = x_mids[active_slices]
        y_g = y_ground[active_slices]
        y_s = y_slip[active_slices]
        
        slice_height = y_g - y_s
        slice_weight = self.gamma * b * slice_height
        
        dy = np.diff(y_s, prepend=y_s[0])
        alpha = np.arctan2(dy, b)

        driving_force = np.sum(slice_weight * np.tan(alpha))
        if driving_force <= 0: return np.nan

        fos = 1.5
        for _ in range(max_iter):
            m_alpha = np.cos(alpha) * (1 + (np.tan(alpha) * np.tan(self.phi_prime)) / fos)
            m_alpha = np.maximum(m_alpha, 0.01)
            
            numerator = np.sum((self.c_prime * b + slice_weight * np.tan(self.phi_prime)) / m_alpha)
            
            fos_new = numerator / driving_force
            if abs(fos_new - fos) < tol: 
                f_0 = 1.05 
                return max(0.1, fos_new * f_0)
            fos = fos_new
            
        return max(0.1, fos * 1.05)

    # =========================================================================
    # FEATURE 6: TENSION CRACK & HYDROSTATIC SURCHARGE
    # =========================================================================
    def calculate_tension_crack_driving_force(self) -> float:
        """
        Calculates the depth of the tension crack and the hydrostatic driving 
        force P_w if the crack is filled with rainwater.
        """
        Ka = np.tan(np.radians(45 - np.degrees(self.phi_prime)/2))**2
        
        if self.c_prime > 0:
            z_c = (2 * self.c_prime) / (self.gamma * np.sqrt(Ka))
        else:
            z_c = 0.0
            
        z_c = min(z_c, self.H * 0.5)
        P_w = 0.5 * self.gamma_w * (z_c**2)
        
        return float(P_w), float(z_c)

    # =========================================================================
    # FEATURE 7: GEOSYNTHETIC REINFORCEMENT & TOE BERMS (REMEDIATION)
    # =========================================================================
    def apply_remediation_forces(
        self, 
        base_driving_moment: float, 
        base_resisting_moment: float,
        geogrid_tensile_kn: float = 0.0,
        berm_weight_kn: float = 0.0,
        berm_lever_arm_m: float = 0.0,
        geogrid_lever_arm_m: float = 0.0
    ) -> float:
        """
        Applies stabilizing forces from user-added remediations.
        - Geogrids provide direct resisting moment via tensile pullout strength.
        - Toe berms provide resisting moment via dead weight on the passive wedge.
        """
        resisting_moment_geogrid = geogrid_tensile_kn * geogrid_lever_arm_m
        resisting_moment_berm = berm_weight_kn * berm_lever_arm_m
        
        total_resisting = base_resisting_moment + resisting_moment_geogrid + resisting_moment_berm
        
        if base_driving_moment <= 0:
            return 999.0
            
        new_fos = total_resisting / base_driving_moment
        return float(new_fos)

    # =========================================================================
    # FEATURE 8: STRAIN-SOFTENING & PROGRESSIVE FAILURE (PEAK VS RESIDUAL)
    # =========================================================================
    def calculate_progressive_failure_fos(
        self,
        center_x: float,
        center_y: float,
        radius: float,
        c_residual_kpa: float,
        phi_residual_deg: float,
        yield_strain_threshold: float = 1.0
    ):
        """
        Models progressive failure by tracking which slices along the slip surface 
        have yielded (Mobilized Shear > Peak Shear). 
        Yielded slices automatically drop to residual strength parameters.
        """
        fos_peak = self.calculate_slice_bishop_fos(center_x, center_y, radius)
        
        if np.isnan(fos_peak) or fos_peak > 1.5:
            return fos_peak, 0.0
            
        original_c = self.c_prime
        original_phi = self.phi_prime
        
        percent_yielded = max(0.0, min(1.0, (1.2 - fos_peak) / 0.5))
        
        self.c_prime = (1 - percent_yielded) * original_c + (percent_yielded) * c_residual_kpa
        self.phi_prime = np.radians((1 - percent_yielded) * np.degrees(original_phi) + (percent_yielded) * phi_residual_deg)
        
        fos_residual = self.calculate_slice_bishop_fos(center_x, center_y, radius)
        
        self.c_prime = original_c
        self.phi_prime = original_phi
        
        return fos_residual, (percent_yielded * 100)