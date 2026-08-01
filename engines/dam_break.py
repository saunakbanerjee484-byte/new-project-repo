"""
engines/dam_break.py

Numerical Simulation of Dam-Break Floods using 1D/2D Saint-Venant Equations.
=============================================================================

Covers two linked engineering workflows used together in the DeltaPulse
platform:

1. Embankment / earth-dam breach hydrograph generation
   (Froehlich 1995/2008 empirical breach geometry + broad-crested weir
   outflow through a growing trapezoidal breach), and

2. Flood-wave routing of that breach outflow (or a classic instantaneous
   dam-break) down a river reach / floodplain using shock-capturing
   finite-volume solutions of the Saint-Venant (shallow water) equations.

Numerical scheme
-----------------
- Finite Volume Method on a structured grid.
- HLL (Harten-Lax-van Leer) approximate Riemann solver for the convective
  flux. HLL is used (rather than a central scheme) because it remains
  stable and non-oscillatory across the wet/dry front and the
  subcritical <-> supercritical transition that a dam-break wave always
  produces -- exactly the regime this project also needs for the
  torrential/supercritical bridge-pier module.
- Explicit time stepping under a CFL condition.
- Manning's friction source handled semi-implicitly to avoid stiffness
  in very shallow water near the wave front.

Governing equations (1D, conservative form)
--------------------------------------------
    dU/dt + dF(U)/dx = S(U)
    U = [h, hu]^T
    F = [hu, hu^2 + 0.5 g h^2]^T
    S = [0, g h (S0 - Sf)]^T
    S0 = -dz_bed/dx            (bed slope, +ve downhill)
    Sf = n^2 |u| u / h^(4/3)   (Manning friction slope)

2D adds the v-momentum equation and is advanced via Strang dimensional
splitting (half x-sweep, full y-sweep, half x-sweep), reusing the same
1D HLL flux along each sweep direction. The 2D solver omits the bed-slope
source term (idealized/near-flat floodplain) -- see class docstring.

References
----------
- Toro, E.F., "Shock-Capturing Methods for Free-Surface Shallow Flows", 2001.
- Ritter, A., "Die Fortpflanzung der Wasserwellen", 1892 (analytical
  dry-bed dam-break solution -- used in tests/test_dam_break.py).
- Froehlich, D.C., "Embankment Dam Breach Parameters and Their
  Uncertainties", J. Hydraulic Engineering, 2008.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

G = 9.81            # m/s^2
DRY_TOL = 1e-6       # depth (m) below which a cell is treated as dry


# ---------------------------------------------------------------------------
# Core Riemann solver
# ---------------------------------------------------------------------------

def hll_flux_arrays(hL, uL, hR, uR, g: float = G):
    """
    Vectorized HLL approximate Riemann solver for the 1D shallow water
    equations, evaluated at every cell interface simultaneously.

    Parameters
    ----------
    hL, uL, hR, uR : arrays of equal shape
        Left/right reconstructed depth and velocity at each interface.

    Returns
    -------
    Fh, Fhu : arrays
        Mass and momentum flux at each interface.
    """
    hL = np.maximum(hL, 0.0)
    hR = np.maximum(hR, 0.0)

    wetL = hL > DRY_TOL
    wetR = hR > DRY_TOL

    cL = np.where(wetL, np.sqrt(g * np.maximum(hL, 0.0)), 0.0)
    cR = np.where(wetR, np.sqrt(g * np.maximum(hR, 0.0)), 0.0)

    # Wave speed estimates (dry-front fallback per Toro Ch. 6)
    sL = np.where(wetL, np.minimum(uL - cL, uR - cR), uR - 2 * cR)
    sR = np.where(wetR, np.maximum(uL + cL, uR + cR), uL + 2 * cL)

    FhL, FhuL = hL * uL, hL * uL ** 2 + 0.5 * g * hL ** 2
    FhR, FhuR = hR * uR, hR * uR ** 2 + 0.5 * g * hR ** 2

    denom = np.where(np.abs(sR - sL) < 1e-12, 1e-12, sR - sL)
    Fh_star = (sR * FhL - sL * FhR + sL * sR * (hR - hL)) / denom
    Fhu_star = (sR * FhuL - sL * FhuR + sL * sR * (hR * uR - hL * uL)) / denom

    Fh = np.where(sL >= 0, FhL, np.where(sR <= 0, FhR, Fh_star))
    Fhu = np.where(sL >= 0, FhuL, np.where(sR <= 0, FhuR, Fhu_star))

    both_dry = (~wetL) & (~wetR)
    Fh = np.where(both_dry, 0.0, Fh)
    Fhu = np.where(both_dry, 0.0, Fhu)
    return Fh, Fhu


# ---------------------------------------------------------------------------
# Embankment / earth-dam breach hydrograph (Froehlich empirical model)
# ---------------------------------------------------------------------------

def froehlich_breach_params(dam_height_m: float, reservoir_volume_m3: float,
                             failure_mode: str = "piping") -> dict:
    """
    Froehlich (1995/2008) empirical regression for breach geometry and
    formation time -- the standard first-pass estimate used in dam-break
    studies when detailed internal-erosion mechanics are not modeled
    (e.g. for DVC-system barrages / earth embankments in the West Bengal
    reach where detailed geotechnical breach data isn't telemetered).

    Parameters
    ----------
    dam_height_m : height of dam/embankment at breach location (m)
    reservoir_volume_m3 : reservoir storage at time of failure (m^3)
    failure_mode : "piping" or "overtopping"

    Returns
    -------
    dict with Bavg (m), side_slope (H:V), tf_hours
    """
    Vw = reservoir_volume_m3
    hb = dam_height_m
    k0 = 1.4 if failure_mode == "overtopping" else 1.0

    Bavg = 0.27 * k0 * (Vw ** 0.32) * (hb ** 0.04)
    tf_hours = 63.2 * np.sqrt(Vw / (G * hb ** 2)) / 3600.0
    side_slope = 1.0  # 1H:1V -- Froehlich's median recommendation

    return {"Bavg": float(Bavg), "side_slope": side_slope, "tf_hours": float(tf_hours)}


def breach_outflow_hydrograph(reservoir_level_m: float, dam_crest_m: float,
                               dam_base_m: float, breach_params: dict,
                               t_end_s: float, dt_s: float = 10.0,
                               storage_area_m2: Optional[float] = None,
                               weir_coeff: float = 1.7):
    """
    Lumped-reservoir breach outflow hydrograph through a linearly-growing
    trapezoidal breach, using the standard broad-crested weir equation
    (NWS/Froehlich simplified breach model).

    If storage_area_m2 is given the reservoir level is drawn down by mass
    balance as the breach discharges (appropriate for small/medium
    reservoirs). If None, the reservoir level is held constant (large
    reservoir relative to breach outflow, e.g. a major barrage pool).

    Returns
    -------
    t : (N,) s
    Q : (N,) m^3/s  -- breach discharge hydrograph, feed this into
                       SaintVenant1D.run(..., inflow_Q=...) to route the
                       flood wave down the receiving river reach.
    h_res : (N,) m  -- reservoir level history
    """
    Bavg = breach_params["Bavg"]
    m = breach_params["side_slope"]
    tf = breach_params["tf_hours"] * 3600.0

    n_steps = int(t_end_s / dt_s) + 1
    t = np.linspace(0, t_end_s, n_steps)
    Q = np.zeros(n_steps)
    h_res = np.full(n_steps, reservoir_level_m)

    level = reservoir_level_m
    for i, ti in enumerate(t):
        frac = min(ti / tf, 1.0) if tf > 0 else 1.0
        B = Bavg * frac                                          # breach bottom width
        z_breach = dam_crest_m - (dam_crest_m - dam_base_m) * frac  # breach invert lowering
        head = max(level - z_breach, 0.0)

        if head > 0:
            Q_rect = weir_coeff * B * head ** 1.5
            Q_tri = (8.0 / 15.0) * np.sqrt(2 * G) * m * head ** 2.5
            Qi = Q_rect + Q_tri
        else:
            Qi = 0.0

        Q[i] = Qi
        h_res[i] = level

        if storage_area_m2 is not None and i < n_steps - 1:
            level = max(level - Qi * dt_s / storage_area_m2, dam_base_m)

    return t, Q, h_res


def breach_inflow_function(t_arr: np.ndarray, Q_arr: np.ndarray) -> Callable[[float], float]:
    """Wrap a breach hydrograph (t, Q) as a callable Q(t) for SaintVenant1D."""
    def _fn(t: float) -> float:
        return float(np.interp(t, t_arr, Q_arr, left=Q_arr[0], right=Q_arr[-1]))
    return _fn


# ---------------------------------------------------------------------------
# 1D Saint-Venant solver
# ---------------------------------------------------------------------------

class SaintVenant1D:
    """
    1D Saint-Venant shallow-water solver for dam-break / embankment-breach
    flood-wave propagation along a river reach.
    """

    def __init__(self, x, bed_elev, manning_n=0.030, width=1.0):
        """
        x : (N,) m, cell-center chainage (must be uniformly spaced)
        bed_elev : (N,) m, bed elevation at each cell
        manning_n : scalar or (N,) array, Manning's roughness
        width : scalar or (N,) array, channel top width (m) -- converts
                the unit-width SVE solution to a discharge estimate
                (Q = h*u*width) for prismatic-channel approximations.
        """
        self.x = np.asarray(x, dtype=float)
        self.n_cells = len(self.x)
        dx = np.diff(self.x)
        assert np.allclose(dx, dx[0]), "grid must be uniformly spaced"
        self.dx = float(dx[0])
        self.z = np.asarray(bed_elev, dtype=float)
        self.n_manning = (np.full(self.n_cells, manning_n) if np.isscalar(manning_n)
                           else np.asarray(manning_n, dtype=float))
        self.width = (np.full(self.n_cells, width) if np.isscalar(width)
                       else np.asarray(width, dtype=float))

        self.h = np.zeros(self.n_cells)
        self.u = np.zeros(self.n_cells)
        self.t = 0.0

    def set_initial_condition(self, h0, u0=None):
        self.h = np.asarray(h0, dtype=float).copy()
        self.u = np.zeros(self.n_cells) if u0 is None else np.asarray(u0, dtype=float).copy()

    @classmethod
    def dam_break_ic(cls, x, bed_elev, dam_index, h_upstream, h_downstream, **kwargs):
        """
        Convenience constructor for the classic two-reservoir dam-break
        initial condition: still water at depth h_upstream behind the dam
        (cells <= dam_index), h_downstream ahead of it.
        """
        model = cls(x, bed_elev, **kwargs)
        h0 = np.where(np.arange(model.n_cells) <= dam_index, h_upstream, h_downstream)
        model.set_initial_condition(h0)
        return model

    def _apply_bcs(self, h, u, t=None, inflow_Q=None):
        if inflow_Q is not None and t is not None:
            h0 = max(h[0], DRY_TOL)
            u_left = inflow_Q(t) / (self.width[0] * h0)
            h_left = h[0]
        else:
            h_left, u_left = h[0], u[0]
        h_ext = np.concatenate(([h_left], h, [h[-1]]))
        u_ext = np.concatenate(([u_left], u, [u[-1]]))
        return h_ext, u_ext

    def _rhs(self, h, u, t, inflow_Q=None):
        h_ext, u_ext = self._apply_bcs(h, u, t, inflow_Q)

        hL, hR = h_ext[:-1], h_ext[1:]
        uL, uR = u_ext[:-1], u_ext[1:]
        Fh, Fhu = hll_flux_arrays(hL, uL, hR, uR)

        dh_dt = -(Fh[1:] - Fh[:-1]) / self.dx
        dhu_dt = -(Fhu[1:] - Fhu[:-1]) / self.dx

        z_ext = np.concatenate(([self.z[0]], self.z, [self.z[-1]]))
        dzdx = (z_ext[2:] - z_ext[:-2]) / (2 * self.dx)
        dhu_dt += -G * h * dzdx

        return dh_dt, dhu_dt

    def _friction_update(self, h, hu, dt):
        """Semi-implicit Manning friction: avoids stiffness in shallow water."""
        u = np.where(h > DRY_TOL, hu / np.maximum(h, DRY_TOL), 0.0)
        denom = 1.0 + G * dt * self.n_manning ** 2 * np.abs(u) / np.maximum(h, DRY_TOL) ** (4 / 3)
        return np.where(h > DRY_TOL, hu / denom, 0.0)

    def _cfl_dt(self, h, u, cfl):
        c = np.sqrt(G * np.maximum(h, 0.0))
        speed = np.abs(u) + c
        max_speed = max(np.max(speed), 1e-6)
        return cfl * self.dx / max_speed

    def run(self, t_end, cfl=0.45, max_dt=5.0, record_every=60.0,
            gauge_indices=None, inflow_Q: Optional[Callable[[float], float]] = None):
        """
        Advance the simulation from t=0 to t_end.

        gauge_indices : list of cell indices to record continuous
                         hydrographs at (e.g. bridge sites, town gauges).
        inflow_Q : optional callable(t) -> m^3/s prescribed discharge at
                   the upstream boundary (use for a breach hydrograph
                   generated by breach_outflow_hydrograph()). If omitted,
                   the upstream boundary is transmissive (classic
                   instantaneous dam-break).

        Returns dict with 't','h','u' snapshot arrays and 'gauges'
        (per-index continuous hydrographs of stage & discharge).
        """
        h, u = self.h.copy(), self.u.copy()
        t = 0.0
        next_record = 0.0
        records_t, records_h, records_u = [], [], []
        gauges = {gi: {"t": [], "h": [], "Q": []} for gi in (gauge_indices or [])}

        while t < t_end:
            dt = min(self._cfl_dt(h, u, cfl), max_dt, t_end - t)
            dh_dt, dhu_dt = self._rhs(h, u, t, inflow_Q)

            hu = h * u
            h_new = np.maximum(h + dt * dh_dt, 0.0)
            hu_new = hu + dt * dhu_dt
            hu_new = self._friction_update(h_new, hu_new, dt)
            u_new = np.where(h_new > DRY_TOL, hu_new / np.maximum(h_new, DRY_TOL), 0.0)

            h, u = h_new, u_new
            t += dt

            if t >= next_record - 1e-9:
                records_t.append(t)
                records_h.append(h.copy())
                records_u.append(u.copy())
                next_record += record_every

            for gi in gauges:
                Q = h[gi] * u[gi] * self.width[gi]
                gauges[gi]["t"].append(t)
                gauges[gi]["h"].append(float(h[gi]))
                gauges[gi]["Q"].append(float(Q))

        self.h, self.u, self.t = h, u, t
        return {
            "t": np.array(records_t),
            "h": np.array(records_h),
            "u": np.array(records_u),
            "gauges": gauges,
        }


# ---------------------------------------------------------------------------
# 2D Saint-Venant solver (structured grid, dimensional splitting)
# ---------------------------------------------------------------------------

class SaintVenant2D:
    """
    2D Saint-Venant solver on a structured rectangular grid, advanced via
    Strang dimensional splitting using the same HLL flux as the 1D model.
    Intended for floodplain / urban inundation-extent mapping downstream
    of a breach (feeds geospatial/inundation_map.py).

    Simplification: bed-slope source term is omitted (assumes a
    near-flat floodplain cell); for sloped terrain, pre-warp the initial
    water-surface elevation or extend _sweep_x/_sweep_y with a bed
    gradient term analogous to the 1D solver.
    """

    def __init__(self, nx, ny, dx, dy, manning_n=0.035):
        self.nx, self.ny = nx, ny
        self.dx, self.dy = dx, dy
        self.n_manning = manning_n
        self.h = np.zeros((ny, nx))
        self.hu = np.zeros((ny, nx))
        self.hv = np.zeros((ny, nx))
        self.t = 0.0

    def set_initial_condition(self, h0):
        self.h = np.asarray(h0, dtype=float).reshape(self.ny, self.nx).copy()
        self.hu[:] = 0.0
        self.hv[:] = 0.0

    def _sweep_x(self, h, hu, hv, dt):
        u = np.where(h > DRY_TOL, hu / np.maximum(h, DRY_TOL), 0.0)
        v = np.where(h > DRY_TOL, hv / np.maximum(h, DRY_TOL), 0.0)
        h_ext = np.pad(h, ((0, 0), (1, 1)), mode="edge")
        u_ext = np.pad(u, ((0, 0), (1, 1)), mode="edge")
        v_ext = np.pad(v, ((0, 0), (1, 1)), mode="edge")

        hL, hR = h_ext[:, :-1], h_ext[:, 1:]
        uL, uR = u_ext[:, :-1], u_ext[:, 1:]
        Fh, Fhu = hll_flux_arrays(hL, uL, hR, uR)
        Fhv = np.where(uL + uR >= 0, hL * v_ext[:, :-1] * uL, hR * v_ext[:, 1:] * uR)

        dh = -(Fh[:, 1:] - Fh[:, :-1]) / self.dx * dt
        dhu = -(Fhu[:, 1:] - Fhu[:, :-1]) / self.dx * dt
        dhv = -(Fhv[:, 1:] - Fhv[:, :-1]) / self.dx * dt
        return h + dh, hu + dhu, hv + dhv

    def _sweep_y(self, h, hu, hv, dt):
        u = np.where(h > DRY_TOL, hu / np.maximum(h, DRY_TOL), 0.0)
        v = np.where(h > DRY_TOL, hv / np.maximum(h, DRY_TOL), 0.0)
        h_ext = np.pad(h, ((1, 1), (0, 0)), mode="edge")
        v_ext = np.pad(v, ((1, 1), (0, 0)), mode="edge")
        u_ext = np.pad(u, ((1, 1), (0, 0)), mode="edge")

        hB, hT = h_ext[:-1, :], h_ext[1:, :]
        vB, vT = v_ext[:-1, :], v_ext[1:, :]
        Fh, Fhv = hll_flux_arrays(hB, vB, hT, vT)
        Fhu = np.where(vB + vT >= 0, hB * u_ext[:-1, :] * vB, hT * u_ext[1:, :] * vT)

        dh = -(Fh[1:, :] - Fh[:-1, :]) / self.dy * dt
        dhv = -(Fhv[1:, :] - Fhv[:-1, :]) / self.dy * dt
        dhu = -(Fhu[1:, :] - Fhu[:-1, :]) / self.dy * dt
        return h + dh, hu + dhu, hv + dhv

    def _friction(self, h, hu, hv, dt):
        u = np.where(h > DRY_TOL, hu / np.maximum(h, DRY_TOL), 0.0)
        v = np.where(h > DRY_TOL, hv / np.maximum(h, DRY_TOL), 0.0)
        speed = np.sqrt(u ** 2 + v ** 2)
        denom = 1.0 + G * dt * self.n_manning ** 2 * speed / np.maximum(h, DRY_TOL) ** (4 / 3)
        return (np.where(h > DRY_TOL, hu / denom, 0.0),
                np.where(h > DRY_TOL, hv / denom, 0.0))

    def _cfl_dt(self, h, hu, hv, cfl):
        u = np.where(h > DRY_TOL, hu / np.maximum(h, DRY_TOL), 0.0)
        v = np.where(h > DRY_TOL, hv / np.maximum(h, DRY_TOL), 0.0)
        c = np.sqrt(G * np.maximum(h, 0.0))
        sx = max(np.max(np.abs(u) + c), 1e-6)
        sy = max(np.max(np.abs(v) + c), 1e-6)
        return cfl * min(self.dx / sx, self.dy / sy)

    def run(self, t_end, cfl=0.4, max_dt=5.0, record_every=30.0):
        h, hu, hv = self.h.copy(), self.hu.copy(), self.hv.copy()
        t = 0.0
        next_record = 0.0
        rec_t, rec_h = [], []
        while t < t_end:
            dt = min(self._cfl_dt(h, hu, hv, cfl), max_dt, t_end - t)
            h, hu, hv = self._sweep_x(h, hu, hv, dt / 2)
            h, hu, hv = self._sweep_y(h, hu, hv, dt)
            h, hu, hv = self._sweep_x(h, hu, hv, dt / 2)
            hu, hv = self._friction(h, hu, hv, dt)
            h = np.maximum(h, 0.0)
            t += dt
            if t >= next_record - 1e-9:
                rec_t.append(t)
                rec_h.append(h.copy())
                next_record += record_every
        self.h, self.hu, self.hv, self.t = h, hu, hv, t
        return {"t": np.array(rec_t), "h": np.array(rec_h)}
