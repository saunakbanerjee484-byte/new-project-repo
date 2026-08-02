"""
engines/monte_carlo_risk.py

Monte Carlo Stochastic Simulation & Breach Risk Quantification Engine.
Performs 500+ stochastic runs on Manning's n, Peak Discharge, and Embankment Cohesion.
"""

import numpy as np

class MonteCarloRiskEngine:
    def __init__(self, base_discharge=1500.0, base_manning=0.035, dam_height=12.0):
        self.Q_mean = base_discharge
        self.n_mean = base_manning
        self.H_dam = dam_height

    def run_simulation(self, iterations=500):
        """
        Executes Monte Carlo stochastic iterations with Gaussian perturbations.
        """
        np.random.seed(42) # For reproducible engineering results
        
        # Stochastic sampling (Normal distributions around baseline parameters)
        simulated_q = np.random.normal(loc=self.Q_mean, scale=self.Q_mean * 0.15, size=iterations)
        simulated_n = np.random.normal(loc=self.n_mean, scale=self.n_mean * 0.10, size=iterations)
        simulated_cohesion = np.random.normal(loc=50.0, scale=12.0, size=iterations) # kPa
        
        breach_count = 0
        max_water_levels = []
        fos_values = []
        
        for i in range(iterations):
            # Dynamic water level response based on randomized flow and roughness
            q_val = max(simulated_q[i], 100.0)
            n_val = max(simulated_n[i], 0.01)
            c_val = simulated_cohesion[i]
            
            # Simplified hydraulic head calculation over dam structure
            head_water = (q_val * n_val)**0.6 + 4.0 
            max_water_levels.append(head_water)
            
            # Factor of Safety (FoS) against sliding/piping incorporating soil cohesion
            fos = (c_val * 0.2 + 8.0) / max(head_water * 0.6, 0.5)
            fos_values.append(fos)
            
            # Breach criteria: If water level exceeds dam height OR FoS drops below 1.0
            if head_water >= self.H_dam or fos < 1.0:
                breach_count += 1
                
        breach_probability = (breach_count / iterations) * 100.0
        
        return {
            "breach_probability": round(breach_probability, 2),
            "mean_max_water_level": round(float(np.mean(max_water_levels)), 2),
            "min_factor_of_safety": round(float(np.min(fos_values)), 2),
            "water_level_distribution": max_water_levels,
            "iterations_run": iterations
        }