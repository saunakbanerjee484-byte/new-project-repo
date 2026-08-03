"""
solver/pde_solvers.py

1D and 2D Richards' Equation Solvers for Transient Unsaturated Flow
===================================================================
"""

import numpy as np
from scipy.integrate import solve_ivp

class Richards1D:
    def __init__(self, soil_model, z_max_m, n_nodes=30):
        self.soil = soil_model
        self.z_max = z_max_m
        self.n_nodes = n_nodes
        self.dz = z_max_m / (n_nodes - 1)
        self.z_grid = np.linspace(0, z_max_m, n_nodes)

    def _pde_rhs(self, t, psi):
        K = self.soil.hydraulic_conductivity(psi)
        C = np.maximum(self.soil.specific_moisture_capacity(psi), 1e-6)
        
        K_mid = np.sqrt(K[:-1] * K[1:])
        dpsi_dz = (psi[1:] - psi[:-1]) / self.dz
        flux = -K_mid * (dpsi_dz + 1.0)
        
        dpsi_dt = np.zeros(self.n_nodes)
        dpsi_dt[1:-1] = -(flux[1:] - flux[:-1]) / (self.dz * C[1:-1])
        
        # Boundaries
        dpsi_dt[0] = 0.0  # Bottom water table
        q_rain = -self.rainfall_flux_mps
        dpsi_dt[-1] = -(flux[-1] - q_rain) / (self.dz * C[-1])
        
        return dpsi_dt

    def simulate(self, initial_psi, t_max_seconds, rainfall_mm_per_hr=0.0):
        self.rainfall_flux_mps = (rainfall_mm_per_hr / 1000.0) / 3600.0
        t_eval = np.linspace(0, t_max_seconds, 5)
        
        solution = solve_ivp(
            fun=self._pde_rhs,
            t_span=(0, t_max_seconds),
            y0=initial_psi,
            method='BDF',
            t_eval=t_eval,
            rtol=1e-3,
            atol=1e-3
        )
        
        theta_profiles = [self.soil.volumetric_water_content(p) for p in solution.y.T]
        return {
            'time_seconds': solution.t,
            'z_grid': self.z_grid,
            'psi_profiles': solution.y.T,
            'theta_profiles': np.array(theta_profiles)
        }


class Richards2D:
    def __init__(self, soil_model, x_max_m, z_max_m, nx=15, nz=25):
        """
        Initializes the 2D spatial grid (x = horizontal, z = vertical).
        Resolution is kept modest (15x25) to ensure responsive web dashboard performance.
        """
        self.soil = soil_model
        self.x_max = x_max_m
        self.z_max = z_max_m
        self.nx = nx
        self.nz = nz
        self.dx = x_max_m / (nx - 1)
        self.dz = z_max_m / (nz - 1)
        
        self.x_grid = np.linspace(0, x_max_m, nx)
        self.z_grid = np.linspace(0, z_max_m, nz)

    def _pde_rhs(self, t, psi_flat):
        # Reshape flat array back to 2D grid (z_rows, x_cols)
        psi = psi_flat.reshape((self.nz, self.nx))
        
        K = self.soil.hydraulic_conductivity(psi)
        C = np.maximum(self.soil.specific_moisture_capacity(psi), 1e-6)
        
        dpsi_dt = np.zeros_like(psi)
        
        # 1. Vertical fluxes (z-direction) -> includes gravity (+1.0)
        K_mid_z = np.sqrt(K[:-1, :] * K[1:, :])
        dpsi_dz = (psi[1:, :] - psi[:-1, :]) / self.dz
        flux_z = -K_mid_z * (dpsi_dz + 1.0)
        
        # 2. Horizontal fluxes (x-direction) -> capillary only, no gravity
        K_mid_x = np.sqrt(K[:, :-1] * K[:, 1:])
        dpsi_dx = (psi[:, 1:] - psi[:, :-1]) / self.dx
        flux_x = -K_mid_x * dpsi_dx
        
        # 3. Internal nodes mass balance
        dflux_z = (flux_z[1:, 1:-1] - flux_z[:-1, 1:-1]) / self.dz
        dflux_x = (flux_x[1:-1, 1:] - flux_x[1:-1, :-1]) / self.dx
        dpsi_dt[1:-1, 1:-1] = -(dflux_z + dflux_x) / C[1:-1, 1:-1]
        
        # 4. Boundary Conditions
        # Bottom (z=0): Fixed water table
        dpsi_dt[0, :] = 0.0
        
        # Sides (x=0, x=L): No-flow boundary (horizontal flux = 0)
        dflux_z_left = (flux_z[1:, 0] - flux_z[:-1, 0]) / self.dz
        dpsi_dt[1:-1, 0] = -(dflux_z_left + (flux_x[1:-1, 0] - 0)/self.dx) / C[1:-1, 0]
        
        dflux_z_right = (flux_z[1:, -1] - flux_z[:-1, -1]) / self.dz
        dpsi_dt[1:-1, -1] = -(dflux_z_right + (0 - flux_x[1:-1, -1])/self.dx) / C[1:-1, -1]
        
        # Top (z=H): Rainfall only applied to the center half of the domain to trigger 2D lateral flow
        q_rain = np.zeros(self.nx)
        mid_idx = self.nx // 2
        spread = self.nx // 4
        q_rain[mid_idx-spread : mid_idx+spread] = -self.rainfall_flux_mps
        
        dflux_x_top = (flux_x[-1, 1:] - flux_x[-1, :-1]) / self.dx
        dpsi_dt[-1, 1:-1] = -((flux_z[-1, 1:-1] - q_rain[1:-1])/self.dz + dflux_x_top) / C[-1, 1:-1]
        dpsi_dt[-1, 0] = -((flux_z[-1, 0] - q_rain[0])/self.dz + (flux_x[-1, 0] - 0)/self.dx) / C[-1, 0]
        dpsi_dt[-1, -1] = -((flux_z[-1, -1] - q_rain[-1])/self.dz + (0 - flux_x[-1, -1])/self.dx) / C[-1, -1]
        
        return dpsi_dt.flatten()

    def simulate(self, initial_psi_2d, t_max_seconds, rainfall_mm_per_hr=0.0):
        self.rainfall_flux_mps = (rainfall_mm_per_hr / 1000.0) / 3600.0
        t_eval = np.array([0, t_max_seconds])
        
        solution = solve_ivp(
            fun=self._pde_rhs,
            t_span=(0, t_max_seconds),
            y0=initial_psi_2d.flatten(),
            method='BDF',
            t_eval=t_eval,
            rtol=1e-2,  # Relaxed slightly for 2D speed in Streamlit
            atol=1e-2
        )
        
        final_psi_2d = solution.y[:, -1].reshape((self.nz, self.nx))
        final_theta_2d = self.soil.volumetric_water_content(final_psi_2d)
        
        return {
            'z_grid': self.z_grid,
            'x_grid': self.x_grid,
            'final_theta_2d': final_theta_2d
        }