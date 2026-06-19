"""Test-sequence state machine + settle-time/error tradeoff."""

import pytest

from leak_test_sim import (
    SequenceConfig, run_sequence, settle_error_tradeoff, ThermalTransient,
    conductance_from_leak_rate, sccm_to_pa_m3_s, Phase,
)


def test_sealed_sequence_zero_leak():
    cfg = SequenceConfig(test_pressure=300000.0, volume=1e-4,
                         settle_time=5.0, test_time=5.0, dt=0.01)
    res = run_sequence(cfg, C=0.0)
    assert res.dP == pytest.approx(0.0, abs=1e-6)
    assert res.leak_rate_si == pytest.approx(0.0, abs=1e-9)


def test_sequence_recovers_known_leak():
    """Full fill->settle->test recovers the known leak rate (no thermal)."""
    cfg = SequenceConfig(test_pressure=250000.0, volume=5e-4,
                         settle_time=5.0, test_time=5.0, dt=0.005)
    Q_true = sccm_to_pa_m3_s(3.0)
    C = conductance_from_leak_rate(Q_true, cfg.test_pressure)
    res = run_sequence(cfg, C=C)
    # measured leak rate matches true within a couple percent (finite window)
    assert res.leak_rate_si == pytest.approx(Q_true, rel=0.02)
    assert res.dP < 0.0
    # measured value should be reported in all units consistently
    u = res.leak_rate_units
    assert u["sccm"] == pytest.approx(res.leak_rate_si / sccm_to_pa_m3_s(1.0))


def test_dp_taken_only_over_test_window():
    """ΔP must reflect the test window length, not settle+test."""
    cfg = SequenceConfig(test_pressure=300000.0, volume=1e-4,
                         settle_time=10.0, test_time=4.0, dt=0.005)
    Q_true = sccm_to_pa_m3_s(5.0)
    C = conductance_from_leak_rate(Q_true, cfg.test_pressure)
    res = run_sequence(cfg, C=C)
    assert res.dt_test == 4.0
    # leak_rate = |dP| * V / test_time, consistent with the reported dP
    assert res.leak_rate_si == pytest.approx(abs(res.dP) * cfg.volume / 4.0)


def test_settle_error_tradeoff_decreases():
    """Longer settle -> smaller thermal-induced false reading (monotone-ish)."""
    cfg = SequenceConfig(test_pressure=300000.0, volume=1e-4,
                         settle_time=5.0, test_time=5.0, dt=0.01)
    thermal = ThermalTransient(T_ambient=293.15, dT0=4.0, tau_thermal=8.0)
    rows = settle_error_tradeoff(cfg, thermal, C=0.0,
                                 settle_times=[1.0, 5.0, 20.0, 60.0])
    leaks = [abs(r["leak_rate_si"]) for r in rows]
    # strictly decreasing false reading as settle grows
    assert leaks[0] > leaks[1] > leaks[2] > leaks[3]
    # at long settle the thermal transient is essentially gone
    assert leaks[-1] < 0.01 * leaks[0]


def test_phase_enum():
    assert Phase.FILL.value == "fill"
    assert {p.value for p in Phase} == {"fill", "settle", "test", "exhaust"}
