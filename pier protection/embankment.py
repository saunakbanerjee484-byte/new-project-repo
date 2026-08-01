import math
import logging
from typing import Dict, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarthDamSeepage:
    """
    Geotechnical & Hydraulic Numerical Engine: Earth Dam Seepage Analysis.
    
    ===========================================================================
    PHYSICS & THEORETICAL OVERVIEW
    ===========================================================================
    This module models unconfined steady-state seepage through a homogeneous 
    earthen dam using the principles of fluid mechanics in porous media.
    
    1. Darcy's Law (Flow in Porous Media):
       Governs the macroscopic flow of water through the soil matrix. The total 
       discharge per unit width is derived from the permeability and focal distance.
       Equation: $q = k \cdot y_0$
       
    2. Casagrande's Base Parabola (1937):
       Arthur Casagrande established that the phreatic line (the top flow line 
       where pore water pressure is strictly atmospheric, u=0) mathematically 
       approximates a parabola. The focus of this base parabola is situated at 
       the downstream toe drain.
       Equation: $y_0 = \sqrt{d^2 + h^2} - d$
       
    3. Terzaghi's Effective Stress & Critical Exit Gradient:
       Geotechnical failure via 'piping' (internal erosion) occurs at the downstream 
       toe when the upward hydrodynamic seepage force completely neutralizes the 
       submerged effective weight of the soil mass, bringing effective stress to zero.
       Equation: $i_c = \frac{G_s - 1}{1 + e}$

    ===========================================================================
    SOURCES & REFERENCES
    ===========================================================================
    - Casagrande, A. (1937). "Seepage Through Dams." Journal of the New England 
      Water Works Association, 51(2), 131-172.
    - Terzaghi, K. (1943). "Theoretical Soil Mechanics." John Wiley and Sons.
    - CWC (Central Water Commission) Guidelines for Earth Dam Design.
    ===========================================================================
    """

    # Casagrande empirical correction table for exit point Delta_a / (a + Delta_a)
    # Mapping downstream slope angle (alpha in degrees) to correction factor.
    CASAGRANDE_CORRECTION_TABLE = {
        30: 0.36,
        60: 0.32,
        90: 0.26,
        120: 0.18,
        150: 0.10,
        180: 0.00
    }

    def __init__(self, permeability_k: float = 1e-5, specific_gravity_gs: float = 2.65, void_ratio_e: float = 0.6):
        """
        Initialize the earth dam parameters.
        
        Args:
            permeability_k (float): Coefficient of permeability (k) in m/s.
            specific_gravity_gs (float): Specific gravity of soil solids (Gs).
            void_ratio_e (float): Void ratio (e) of the embankment soil matrix.
        """
        self.k = permeability_k
        self.Gs = specific_gravity_gs
        self.e = void_ratio_e

    def calculate_base_parabola(self, d: float, h: float) -> float:
        """
        Calculates the focal distance (y_0) of the Casagrande base parabola.
        
        Physics Context: Defines the fundamental geometry of the flow net's top boundary.
        
        Args:
            d (float): Horizontal distance from the focus to the upstream water surface entry point (m).
            h (float): Head of water on the upstream side (m).
            
        Returns:
            float: y_0 parameter of the base parabola (m).
        """
        try:
            y0 = math.sqrt(d**2 + h**2) - d
            return y0
        except ValueError as err:
            logger.error(f"Invalid geometry for base parabola calculation: {err}")
            return 0.0

    def calculate_seepage_discharge(self, y0: float) -> float:
        """
        Calculates the steady-state seepage discharge per unit width of the dam.
        
        Physics Context: Application of Darcy's Law ($v = k \cdot i$) integrated over the flow area.
        
        Args:
            y0 (float): Focal distance of the base parabola (m).
            
        Returns:
            float: Discharge q (m^3/s per meter width).
        """
        return self.k * y0

    def _get_casagrande_correction_factor(self, alpha_deg: float) -> float:
        """
        Interpolates the Casagrande empirical correction factor based on downstream slope angle.
        Used to correct the theoretical parabolic breakout point to the actual physical breakout point.
        """
        angles = sorted(self.CASAGRANDE_CORRECTION_TABLE.keys())
        
        if alpha_deg in self.CASAGRANDE_CORRECTION_TABLE:
            return self.CASAGRANDE_CORRECTION_TABLE[alpha_deg]
            
        if alpha_deg <= angles[0]:
            return self.CASAGRANDE_CORRECTION_TABLE[angles[0]]
        if alpha_deg >= angles[-1]:
            return self.CASAGRANDE_CORRECTION_TABLE[angles[-1]]
            
        for i in range(len(angles) - 1):
            a1, a2 = angles[i], angles[i+1]
            if a1 < alpha_deg < a2:
                c1 = self.CASAGRANDE_CORRECTION_TABLE[a1]
                c2 = self.CASAGRANDE_CORRECTION_TABLE[a2]
                return c1 + (c2 - c1) * ((alpha_deg - a1) / (a2 - a1))
        
        return 0.0

    def calculate_phreatic_exit(self, y0: float, alpha_deg: float) -> Dict[str, float]:
        """
        Calculates the actual exit breakout point of the phreatic line on the downstream face.
        
        Physics Context: The theoretical base parabola exits the downstream face outside the dam.
        Casagrande's graphical correction applies $\\Delta a$ to pull the breakout point 'a' 
        inward, ensuring it tangentially intersects the discharge face.
        
        Args:
            y0 (float): Focal distance of the base parabola (m).
            alpha_deg (float): Downstream slope angle in degrees.
            
        Returns:
            dict: Breakout distance 'a' and correction '\\Delta a'.
        """
        correction_factor = self._get_casagrande_correction_factor(alpha_deg)
        alpha_rad = math.radians(alpha_deg)
        
        try:
            a_plus_delta_a = y0 / (1 - math.cos(alpha_rad))
            delta_a = a_plus_delta_a * correction_factor
            a = a_plus_delta_a - delta_a
            
            return {
                "a_plus_delta_a": round(a_plus_delta_a, 4),
                "delta_a": round(delta_a, 4),
                "actual_breakout_a": round(a, 4)
            }
        except ZeroDivisionError:
            logger.error("Slope angle alpha cannot be 0 degrees.")
            return {"a_plus_delta_a": 0.0, "delta_a": 0.0, "actual_breakout_a": 0.0}

    def calculate_critical_gradient(self) -> float:
        """
        Calculates the critical exit gradient ($i_c$) to prevent piping failure.
        
        Physics Context: Derived from Terzaghi's effective stress. When the upward hydraulic 
        gradient 'i' reaches $i_c$, effective stress becomes zero (quick condition).
        
        Returns:
            float: Critical exit gradient.
        """
        return (self.Gs - 1) / (1 + self.e)

    def analyze_embankment(self, d: float, h: float, alpha_deg: float, actual_exit_gradient: float = None) -> Dict[str, Union[float, dict]]:
        """
        Executes a comprehensive hydrodynamic and geotechnical seepage analysis.
        """
        y0 = self.calculate_base_parabola(d, h)
        discharge = self.calculate_seepage_discharge(y0)
        exit_geometry = self.calculate_phreatic_exit(y0, alpha_deg)
        ic = self.calculate_critical_gradient()
        
        report = {
            "y0_base_parabola_m": round(y0, 4),
            "discharge_q_m3s_m": discharge,
            "exit_geometry_m": exit_geometry,
            "critical_gradient_ic": round(ic, 4)
        }
        
        if actual_exit_gradient:
            fos = ic / actual_exit_gradient if actual_exit_gradient > 0 else float('inf')
            report["factor_of_safety_piping"] = round(fos, 3)
            report["is_safe"] = fos >= 1.5  # Standard FoS for earth dams is generally > 1.5
            
        return report

# Instantiate engine for global usage
embankment_engine = EarthDamSeepage()