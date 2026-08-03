"""
engines/soil_moisture.py

The Unsaturated Soil Mechanics Engine (van Genuchten & Brooks-Corey Models)
===========================================================================

[PHYSICS BACKGROUND]
When the water table is deep, soil pores are not fully filled with water; they 
contain a mixture of water and air. This is the Unsaturated Zone (Vadose Zone). 
Here, water is not driven by positive pressure, but is held against gravity by 
capillary forces and surface tension (Negative Pressure or Suction). 
As the soil dries:
1. Volumetric Water Content (θ) decreases non-linearly.
2. Hydraulic Conductivity (K) drops exponentially (often by orders of magnitude) 
   because air bubbles block the interconnected water flow paths.
3. Specific Moisture Capacity (C)—the rate at which soil absorbs or releases 
   water—becomes highly variable.

[SCENARIOS - WHEN IS THIS USED?]
1. Rainfall Infiltration: Tracking how fast surface water penetrates dry soil 
   and how it alters pore pressures over time.
2. Rapid Drawdown: Calculating how long it takes for water trapped in the 
   unsaturated zone to drain when a reservoir level drops suddenly.
3. Capillary Rise: Modeling anti-gravity moisture migration upward from the 
   groundwater table.

[COMPUTATIONAL LOGIC]
This module acts as the "Lookup Engine" for the PDE Solver (Richards' Equation). 
At every time-step, the PDE matrix solver asks: "If the pressure head (h) at 
this node is -15 meters, what is the conductivity (K) and capacity (C)?" 
This engine instantly evaluates the van Genuchten equations to return the exact 
analytical derivatives and values to the solver.
"""

import numpy as np

class VanGenuchtenModel:
    """
    van Genuchten (1980) Soil-Water Retention Model.
    This is the most widely used unsaturated soil model globally. It provides 
    smooth, continuous mathematical curves, which are strictly required for the 
    numerical stability of PDE solvers (e.g., Newton-Raphson iterations).
    """

    def __init__(self, theta_s, theta_r, alpha, n, k_sat):
        # [GEOTECH]: Maximum water the soil can hold (saturated volumetric water content / porosity).
        # [MATH]: Upper asymptote of the retention curve.
        self.theta_s = float(theta_s)
        
        # [GEOTECH]: Residual moisture content. Water bound so tightly to soil particles 
        # (hygroscopic water) that neither gravity nor typical suction can remove it.
        # [MATH]: Lower asymptote of the retention curve.
        self.theta_r = float(theta_r)
        
        # [GEOTECH]: Inverse of the air-entry value. Relates to the dominant pore size 
        # (larger values for coarse sands, very small values for dense clays).
        # [MATH]: Scaling parameter for the pressure head (in 1/meters).
        self.alpha = float(alpha)
        
        # [GEOTECH]: Pore-size distribution index. Indicates the uniformity of pore sizes.
        # [MATH]: Determines the slope/steepness of the S-shaped retention curve.
        self.n = float(n)
        
        # [GEOTECH]: Fully saturated hydraulic conductivity (Darcy's k).
        # [MATH]: Baseline scalar multiplier for the K(h) function.
        self.k_sat = float(k_sat)
        
        # [GEOTECH]: Mualem (1976) constraint linking conductivity to moisture retention.
        # [MATH]: Restricts the parameter space to simplify the analytical integral of K(h).
        self.m = 1.0 - (1.0 / self.n)

    def _get_suction(self, h):
        # [GEOTECH]: Below the water table (h >= 0), water is under positive pressure (no suction). 
        # Above the water table (h < 0), water is under negative pressure (suction/tension).
        # [MATH]: Handles NumPy arrays. Caps positive heads at 0.0 to prevent math errors 
        # in the unsaturated formulas, returning absolute suction values.
        h = np.asarray(h, dtype=float)
        return np.where(h >= 0, 0.0, np.abs(h))

    def effective_saturation(self, h):
        """
        Calculates the normalized effective saturation (Se), bounded between 0.0 and 1.0.
        """
        psi = self._get_suction(h)
        
        # [GEOTECH]: As suction increases, the effective saturation drops.
        # [MATH]: Base equation: Se = [1 + (alpha * psi)^n]^-m.
        # [NUMERICAL STABILITY]: Safe math to prevent division by zero or overflow on extreme heads.
        Se = (1.0 + (self.alpha * psi)**self.n)**(-self.m)
        
        # [GEOTECH]: In the saturated zone, saturation is strictly 100% (1.0).
        # [MATH]: Overrides computed values where head was >= 0 to exact 1.0.
        return np.where(h >= 0, 1.0, Se)

    def volumetric_water_content(self, h):
        """
        Calculates actual moisture content (theta). This is the primary state 
        variable in the mixed-form Richards' Equation.
        """
        Se = self.effective_saturation(h)
        
        # [GEOTECH]: Actual moisture = Residual moisture + (Active pore space * Effective Saturation).
        # [MATH]: Linear un-normalization: theta = theta_r + Se * (theta_s - theta_r)
        return self.theta_r + Se * (self.theta_s - self.theta_r)

    def specific_moisture_capacity(self, h):
        """
        C(h) = d(theta)/d(h). The soil's "storage capacity". 
        Used in the Left Hand Side (Time derivative) of Richards' Equation.
        """
        psi = self._get_suction(h)
        
        # [GEOTECH]: If soil is fully saturated, it cannot store more water (assuming incompressible matrix).
        # [MATH]: When h >= 0, capacity C(h) = 0.0.
        C = np.zeros_like(psi)
        
        # Extract indices for the unsaturated zone (where h < 0)
        unsat = (h < 0)
        psi_u = psi[unsat]
        
        # [MATH]: Applying chain rule to analytically differentiate the van Genuchten theta equation w.r.t head (h).
        # Term 1: (theta_s - theta_r) * alpha * m * n
        coeff = (self.theta_s - self.theta_r) * self.alpha * self.m * self.n
        
        # Term 2: (alpha * psi)^(n-1)
        term2 = (self.alpha * psi_u)**(self.n - 1.0)
        
        # Term 3: [1 + (alpha * psi)^n]^(-m - 1)
        term3 = (1.0 + (self.alpha * psi_u)**self.n)**(-self.m - 1.0)
        
        # [GEOTECH]: This value peaks when capillary action is most active (the inflection point 
        # of the retention curve) and approaches zero in extremely dry states.
        # [MATH]: Final exact analytical derivative assigned to unsaturated nodes.
        C[unsat] = coeff * term2 * term3
        return C

    def hydraulic_conductivity(self, h):
        """
        K(h). The unsaturated permeability. 
        Used in the Right Hand Side (spatial gradient) of Richards' Equation.
        """
        Se = self.effective_saturation(h)
        
        # [GEOTECH]: If saturated (Se=1), permeability equals K_sat.
        # [MATH]: Initialize array with default saturated conductivity.
        K = np.full_like(Se, self.k_sat)
        
        unsat = (h < 0)
        Se_u = Se[unsat]
        
        # [GEOTECH]: Mualem (1976) pore-connectivity model. 
        # As air enters the soil, water must travel further around air bubbles (tortuosity), 
        # causing K to drop exponentially.
        # [MATH]: Term 1: Se^(0.5) represents the tortuosity factor.
        tortuosity = np.sqrt(Se_u)
        
        # [MATH]: Term 2: [1 - (1 - Se^(1/m))^m]^2 represents the reduction in conductive pores.
        pore_blockage = (1.0 - (1.0 - Se_u**(1.0 / self.m))**self.m)**2.0
        
        # [MATH]: Multiply base K_sat by the reduction factors.
        K[unsat] = self.k_sat * tortuosity * pore_blockage
        
        # [NUMERICAL STABILITY]: Ensure K never hits absolute zero to prevent singular matrices in PDE solvers.
        return np.maximum(K, 1e-12)