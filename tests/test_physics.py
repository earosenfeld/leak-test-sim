"""Pressure-decay physics: closed-form validation.

These are the load-bearing tests. They assert that the simulator reproduces the
analytic leak-test relationships exactly:

  * tau = V / C
  * exponential decay of gauge pressure
  * Q = C*(P - P_atm)  <->  C = Q/(P - P_atm)   (round trip)
  * measured Q = |dP|*V/dt recovers the known Q within tolerance
  * the ODE integrator matches the closed-form solution
"""

import math

import numpy as np
import pytest

from leak_test_sim import (
    ATM, conductance_from_leak_rate, leak_rate_from_conductance, time_constant,
    pressure_decay, delta_p_over_window, leak_rate_from_decay,
    leak_rate_exact_window, simulate_decay, sccm_to_pa_m3_s,
)


def test_time_constant():
    V = 1e-4          # 100 cc
    C = 5e-9          # m^3/s
    assert time_constant(V, C) == pytest.approx(V / C)
    assert time_constant(V, C) == pytest.approx(20000.0)


def test_conductance_leakrate_roundtrip():
    P = 300000.0      # 3 bar absolute
    Q = sccm_to_pa_m3_s(5.0)
    C = conductance_from_leak_rate(Q, P, ATM)
    assert leak_rate_from_conductance(C, P, ATM) == pytest.approx(Q, rel=1e-12)


def test_pressure_decay_closed_form():
    P0 = 300000.0
    V = 1e-4
    C = 1e-8
    tau = V / C
    t = np.array([0.0, tau, 2 * tau, 5 * tau])
    P = pressure_decay(t, P0, V, C, ATM)
    # at t=0, P=P0
    assert P[0] == pytest.approx(P0)
    # gauge pressure decays by e each tau
    g0 = P0 - ATM
    assert (P[1] - ATM) == pytest.approx(g0 * math.exp(-1), rel=1e-12)
    assert (P[2] - ATM) == pytest.approx(g0 * math.exp(-2), rel=1e-12)
    assert (P[3] - ATM) == pytest.approx(g0 * math.exp(-5), rel=1e-12)


def test_measured_leak_rate_recovers_Q():
    """Core validation: known V & C -> simulated dP over dt recovers Q=dP*V/dt.

    For a short window (dt << tau) the measured Q must match C*(P0-P_atm) closely.
    """
    P0 = 250000.0
    V = 5e-4          # 500 cc
    Q_true = sccm_to_pa_m3_s(2.0)
    C = conductance_from_leak_rate(Q_true, P0, ATM)
    tau = V / C
    dt = 5.0
    assert dt < tau / 100.0          # genuinely short window

    dP = delta_p_over_window(P0, V, C, dt, ATM)
    Q_meas = leak_rate_from_decay(dP, V, dt)
    # within 1% for this short window
    assert Q_meas == pytest.approx(Q_true, rel=0.01)
    # and the exact-window helper agrees
    assert leak_rate_exact_window(P0, V, C, dt, ATM) == pytest.approx(Q_meas)


def test_measured_leak_rate_underestimates_for_finite_window():
    """Exponential decay => measured average slope is just below initial Q."""
    P0 = 200000.0
    V = 1e-4
    Q_true = sccm_to_pa_m3_s(10.0)
    C = conductance_from_leak_rate(Q_true, P0, ATM)
    dt = 10.0
    Q_meas = leak_rate_exact_window(P0, V, C, dt, ATM)
    assert Q_meas < Q_true            # under-estimate
    assert Q_meas == pytest.approx(Q_true, rel=0.05)


def test_ode_matches_closed_form_no_temp():
    """The integrator reproduces the analytic exponential decay."""
    P0 = 300000.0
    V = 2e-4
    C = 5e-8
    duration = 30.0
    res = simulate_decay(P0, V, C, duration, dt=0.001, P_atm=ATM)
    P_analytic = pressure_decay(res.t, P0, V, C, ATM)
    # tight agreement everywhere
    assert np.max(np.abs(res.P - P_analytic)) < 1.0   # < 1 Pa over whole trace
    assert res.tau == pytest.approx(V / C)


def test_sealed_part_no_leak():
    """C=0 -> pressure is flat (perfect part)."""
    P0 = 300000.0
    V = 1e-4
    res = simulate_decay(P0, V, C=0.0, duration=20.0, dt=0.01)
    assert np.allclose(res.P, P0)
    assert res.dp(5.0, 15.0) == pytest.approx(0.0, abs=1e-9)


def test_dp_sign_negative_for_leak():
    P0 = 300000.0
    V = 1e-4
    C = 1e-8
    dP = delta_p_over_window(P0, V, C, dt=5.0)
    assert dP < 0.0                    # pressure falls


def test_decayresult_dp_interpolation():
    P0 = 300000.0
    V = 1e-4
    C = 1e-8
    res = simulate_decay(P0, V, C, duration=20.0, dt=0.01)
    dP = res.dp(0.0, 10.0)
    analytic = float(pressure_decay(10.0, P0, V, C)) - P0
    assert dP == pytest.approx(analytic, rel=1e-3)
