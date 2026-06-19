"""Temperature transient + compensation validation.

The headline test: a pure thermal transient with ZERO real leak must make an
uncompensated reading flag a leak, while compensation drives it to ~zero.
"""

import numpy as np
import pytest

from leak_test_sim import (
    ATM, SequenceConfig, run_sequence, ThermalTransient, compensate,
    apparent_dp_from_dt, temperature_corrected_pressure, sccm_to_pa_m3_s,
    DecisionConfig, decide, Verdict, apparent_leak_from_thermal,
)


def test_ideal_gas_apparent_dp():
    # dP = P * dT / T
    P, T, dT = 300000.0, 293.15, -1.0
    assert apparent_dp_from_dt(P, T, dT) == pytest.approx(P * dT / T)
    # 1 K cool-down on 3 bar @ 293 K ~ -1023 Pa
    assert apparent_dp_from_dt(P, T, dT) == pytest.approx(-1023.4, abs=1.0)


def test_compensation_identity_at_tref():
    P, T = 300000.0, 293.15
    assert temperature_corrected_pressure(P, T, T) == pytest.approx(P)


def test_compensation_removes_thermal_offset():
    # If P scaled purely by T/T_ref (ideal gas), compensation recovers P_ref.
    T_ref = 293.15
    T = 295.0
    P_ref = 300000.0
    P_observed = P_ref * (T / T_ref)          # thermal inflation, no leak
    assert compensate(P_observed, T, T_ref) == pytest.approx(P_ref, rel=1e-12)


def test_pure_thermal_transient_false_reject_then_compensated():
    """THE differentiator test.

    Zero real leak (C=0), but the gas cools through the test window. Uncompensated
    -> apparent leak above reject limit (false reject). Compensated -> ~zero.
    """
    cfg = SequenceConfig(
        test_pressure=300000.0, volume=1e-4,
        fill_time=3.0, settle_time=2.0, test_time=5.0, exhaust_time=2.0, dt=0.005,
    )
    # fill heating: starts 4 K hot, relaxes with tau=8 s -> still cooling in test
    thermal = ThermalTransient(T_ambient=293.15, dT0=4.0, tau_thermal=8.0)
    reject_limit = sccm_to_pa_m3_s(1.0)       # 1 sccm reject

    res = run_sequence(cfg, C=0.0, temp_profile=thermal, T_ref=293.15)

    # Uncompensated: there IS an apparent leak, and it should trip the limit.
    assert abs(res.leak_rate_si) > 0.0
    d_uncomp = decide(abs(res.leak_rate_si), DecisionConfig(reject_limit=reject_limit))
    assert d_uncomp.verdict is Verdict.REJECT, (
        f"expected false reject, got {d_uncomp.verdict} "
        f"(uncomp leak={res.leak_rate_si:.3e})")

    # Compensated: apparent leak collapses toward zero, well under the limit.
    assert res.leak_rate_si_compensated is not None
    assert abs(res.leak_rate_si_compensated) < 0.01 * abs(res.leak_rate_si)
    d_comp = decide(abs(res.leak_rate_si_compensated),
                    DecisionConfig(reject_limit=reject_limit))
    assert d_comp.verdict is Verdict.ACCEPT


def test_compensation_preserves_real_leak():
    """Compensation must NOT erase a genuine leak (only the thermal part)."""
    cfg = SequenceConfig(
        test_pressure=300000.0, volume=1e-4,
        fill_time=3.0, settle_time=2.0, test_time=5.0, dt=0.005,
    )
    Q_true = sccm_to_pa_m3_s(5.0)
    from leak_test_sim import conductance_from_leak_rate
    C = conductance_from_leak_rate(Q_true, cfg.test_pressure)
    thermal = ThermalTransient(T_ambient=293.15, dT0=3.0, tau_thermal=8.0)

    res = run_sequence(cfg, C=C, temp_profile=thermal, T_ref=293.15)

    # Compensated reading should recover the true leak (thermal stripped out),
    # while uncompensated is biased by the cooling.
    assert res.leak_rate_si_compensated == pytest.approx(Q_true, rel=0.05)
    # uncompensated is off by more than the compensated one
    err_uncomp = abs(res.leak_rate_si - Q_true)
    err_comp = abs(res.leak_rate_si_compensated - Q_true)
    assert err_comp < err_uncomp


def test_apparent_leak_from_thermal_helper():
    V, P, T, dT, dt = 1e-4, 300000.0, 293.15, -0.5, 5.0
    q = apparent_leak_from_thermal(V, P, T, dT, dt)
    expected = abs(P * dT / T) * V / dt
    assert q == pytest.approx(expected)


def test_thermal_transient_profile():
    th = ThermalTransient(T_ambient=293.15, dT0=5.0, tau_thermal=10.0)
    assert th(0.0) == pytest.approx(298.15)
    assert th(10.0) == pytest.approx(293.15 + 5.0 * np.exp(-1))
    assert th(1e6) == pytest.approx(293.15, abs=1e-6)
