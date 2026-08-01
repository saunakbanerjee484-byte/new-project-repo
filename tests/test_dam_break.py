"""
tests/test_dam_break.py

Verifies the HLL-based SaintVenant1D solver against Ritter's (1892)
closed-form analytical solution for an idealized dam-break on a dry,
frictionless, horizontal bed -- the standard benchmark for dam-break
numerical codes.
"""

import numpy as np
import pytest

from engines.dam_break import SaintVenant1D, G


def ritter_solution(x, t, h0):
    """Analytical dry-bed dam-break solution (Ritter, 1892), dam at x=0."""
    c0 = np.sqrt(G * h0)
    h = np.zeros_like(x)
    u = np.zeros_like(x)
    xi = x / t

    left = xi <= -c0
    fan = (xi > -c0) & (xi < 2 * c0)
    dry = xi >= 2 * c0

    h[left] = h0
    u[left] = 0.0

    c_fan = (1.0 / 3.0) * (2 * c0 - xi[fan])
    u[fan] = (2.0 / 3.0) * (c0 + xi[fan])
    h[fan] = c_fan ** 2 / G

    h[dry] = 0.0
    u[dry] = 0.0
    return h, u


def test_dry_bed_dam_break_matches_ritter():
    # First-order HLL/Godunov schemes are known to be diffusive right at
    # a dry wetting front (Toro 2001, Ch. 6/13), so the leading-edge
    # position lags the inviscid Ritter front somewhat -- the tolerance
    # below reflects that well-documented, expected first-order behavior,
    # not an implementation bug. A MUSCL/2nd-order reconstruction would
    # tighten this considerably.
    L, N = 4000.0, 2000
    x = np.linspace(-L / 2, L / 2, N)
    bed = np.zeros(N)
    dam_index = N // 2
    h0 = 10.0

    model = SaintVenant1D.dam_break_ic(
        x, bed, dam_index, h_upstream=h0, h_downstream=0.0, manning_n=0.0
    )
    t_end = 20.0
    model.run(t_end, cfl=0.4, record_every=t_end)

    h_num, u_num = model.h, model.u
    h_ana, _ = ritter_solution(x, t_end, h0)

    front = 2 * np.sqrt(G * h0) * t_end
    interior = (np.abs(x) < front - 50) & (np.abs(x) > 30)

    err_h = np.abs(h_num[interior] - h_ana[interior])
    assert np.mean(err_h) < 0.35, f"mean depth error too high: {np.mean(err_h):.3f} m"

    wetted = x[h_num > 1e-3]
    front_num = wetted.max() if len(wetted) else 0
    assert abs(front_num - front) / front < 0.20, "wave-front position off by >20%"


def test_still_water_stays_still():
    """A flat, level, frictionless channel with no dam should stay at rest (well-balancedness sanity check)."""
    x = np.linspace(0, 1000, 200)
    bed = np.zeros(200)
    model = SaintVenant1D(x, bed, manning_n=0.0)
    model.set_initial_condition(h0=np.full(200, 3.0))
    model.run(60.0, record_every=60.0)
    assert np.allclose(model.h, 3.0, atol=1e-2)
    assert np.allclose(model.u, 0.0, atol=1e-2)


def test_mass_is_conserved_before_wave_reaches_boundary():
    x = np.linspace(-1000, 1000, 400)
    bed = np.zeros(400)
    model = SaintVenant1D.dam_break_ic(x, bed, 200, h_upstream=5.0, h_downstream=1.0, manning_n=0.0)
    dx = x[1] - x[0]
    mass0 = np.sum(model.h) * dx
    model.run(10.0, record_every=10.0)  # wave front stays well inside domain
    mass1 = np.sum(model.h) * dx
    assert abs(mass1 - mass0) / mass0 < 0.01
