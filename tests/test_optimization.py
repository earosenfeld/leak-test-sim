"""Cycle-time optimization: uncertainty model, optimizer, and SPRT.

These cross-check the settle-time-vs-error tradeoff against its closed forms:
the systematic term must decay with settle, the noise term with test, the RSS
total must be monotone toward the floor, the optimizer must return a near-minimal
cycle that meets the target, and the SPRT must terminate early with the right
call on clearly-good / clearly-bad parts.
"""

import numpy as np
import pytest

from leak_test_sim import (
    UncertaintyModel, measurement_uncertainty, systematic_error, noise_error,
    optimize_cycle_time, sequential_decision, SPRTDecision,
    noise_to_leak_rate_sigma, sccm_to_pa_m3_s, conductance_from_leak_rate,
    ThermalTransient,
)


# -----------------------------------------------------------------------------
# Uncertainty model: monotonicity toward a floor
# -----------------------------------------------------------------------------

def test_uncertainty_decreases_with_settle_time():
    """Longer SETTLE -> smaller systematic term -> smaller total uncertainty."""
    m = UncertaintyModel()
    settles = [0.0, 2.0, 5.0, 10.0, 20.0, 40.0]
    u = [measurement_uncertainty(s, test_t=5.0, model=m) for s in settles]
    # strictly decreasing across the whole range
    assert all(u[i] > u[i + 1] for i in range(len(u) - 1))


def test_uncertainty_decreases_with_test_time():
    """Longer TEST -> smaller noise term -> smaller total uncertainty.

    Settle is held long enough that the systematic term is negligible, isolating
    the test-time (noise) dependence.
    """
    m = UncertaintyModel()
    tests = [1.0, 2.0, 5.0, 10.0, 20.0, 40.0]
    u = [measurement_uncertainty(settle_t=40.0, test_t=t, model=m) for t in tests]
    assert all(u[i] > u[i + 1] for i in range(len(u) - 1))


def test_systematic_term_decays_exponentially_with_settle():
    """Systematic error must fall ~exp(-settle/tau_thermal)."""
    m = UncertaintyModel(tau_thermal=8.0)
    e0 = systematic_error(0.0, test_t=5.0, model=m)
    # one thermal time constant later -> reduced by ~1/e
    e_tau = systematic_error(8.0, test_t=5.0, model=m)
    assert e_tau == pytest.approx(e0 * np.exp(-1.0), rel=1e-6)


def test_noise_term_matches_gage_rr_closed_form():
    """The noise component equals the gauge repeatability sqrt(2)*sigma*V/test."""
    m = UncertaintyModel(sigma_P=5.0, volume=1e-4)
    for t in (2.0, 5.0, 10.0):
        assert noise_error(t, model=m) == pytest.approx(
            noise_to_leak_rate_sigma(m.sigma_P, m.volume, t))


def test_total_is_rss_of_terms():
    """Total uncertainty is the RSS of systematic, noise, and floor."""
    m = UncertaintyModel(noise_floor=sccm_to_pa_m3_s(0.05))
    s, t = 6.0, 4.0
    e_sys = systematic_error(s, t, model=m)
    e_noise = noise_error(t, model=m)
    expected = np.sqrt(e_sys ** 2 + e_noise ** 2 + m.noise_floor ** 2)
    assert measurement_uncertainty(s, t, model=m) == pytest.approx(expected)


def test_uncertainty_floor_is_approached_not_crossed():
    """With long settle+test the total collapses toward the noise floor."""
    floor = sccm_to_pa_m3_s(0.1)
    m = UncertaintyModel(noise_floor=floor)
    u = measurement_uncertainty(settle_t=60.0, test_t=60.0, model=m)
    # essentially at the floor, and never below it
    assert u >= floor
    assert u == pytest.approx(floor, rel=0.05)


# -----------------------------------------------------------------------------
# Optimizer: meets target and is near-minimal
# -----------------------------------------------------------------------------

def test_optimize_meets_target():
    """Recommended cycle's modelled uncertainty is <= the target."""
    m = UncertaintyModel()
    target = sccm_to_pa_m3_s(0.5)
    rec = optimize_cycle_time(target, m)
    assert rec.feasible and rec.met
    assert rec.uncertainty <= target
    # the reported uncertainty is self-consistent with the model at those times
    assert measurement_uncertainty(rec.settle_t, rec.test_t, model=m) == \
        pytest.approx(rec.uncertainty, rel=1e-9)


def test_optimize_is_near_minimal():
    """Shrinking the recommended cycle should break the target.

    Cutting both settle and test below the recommendation must push the
    uncertainty back above target -- proving the recommended cycle is near the
    minimum, not gratuitously long.
    """
    m = UncertaintyModel()
    target = sccm_to_pa_m3_s(0.5)
    rec = optimize_cycle_time(target, m)
    shorter = measurement_uncertainty(
        settle_t=max(rec.settle_t - 3.0, 0.0),
        test_t=max(rec.test_t - 1.0, 0.5), model=m)
    assert shorter > target


def test_optimize_tighter_target_costs_more_time():
    """A tighter uncertainty target requires a longer total cycle."""
    m = UncertaintyModel()
    loose = optimize_cycle_time(sccm_to_pa_m3_s(1.0), m)
    tight = optimize_cycle_time(sccm_to_pa_m3_s(0.3), m)
    assert tight.total_t > loose.total_t
    assert tight.uncertainty < loose.uncertainty


def test_optimize_infeasible_below_floor():
    """A target below the irreducible floor is flagged infeasible."""
    floor = sccm_to_pa_m3_s(0.2)
    m = UncertaintyModel(noise_floor=floor)
    rec = optimize_cycle_time(sccm_to_pa_m3_s(0.1), m)   # below the 0.2 floor
    assert not rec.feasible
    assert not rec.met
    # returns the closest-achievable (near-floor) point
    assert rec.uncertainty == pytest.approx(floor, rel=0.05)


# -----------------------------------------------------------------------------
# SPRT: early termination with correct decisions
# -----------------------------------------------------------------------------

def _fixed_window_samples(max_test_time, sample_dt):
    return int(round(max_test_time / sample_dt))


def test_sprt_good_part_accepts_early():
    """A clearly-good part is ACCEPTED in fewer samples than the fixed window."""
    m = UncertaintyModel()
    reject_limit = sccm_to_pa_m3_s(5.0)
    C_good = conductance_from_leak_rate(sccm_to_pa_m3_s(1.0), m.test_pressure)
    max_t, dt = 30.0, 0.1
    res = sequential_decision(reject_limit, m, C=C_good,
                              max_test_time=max_t, sample_dt=dt, seed=7)
    assert res.decision is SPRTDecision.ACCEPT
    assert res.accepted
    assert res.n_samples < _fixed_window_samples(max_t, dt)
    assert res.test_time < max_t


def test_sprt_bad_part_rejects_early():
    """A clearly-bad part is REJECTED in fewer samples than the fixed window."""
    m = UncertaintyModel()
    reject_limit = sccm_to_pa_m3_s(5.0)
    C_bad = conductance_from_leak_rate(sccm_to_pa_m3_s(10.0), m.test_pressure)
    max_t, dt = 30.0, 0.1
    res = sequential_decision(reject_limit, m, C=C_bad,
                              max_test_time=max_t, sample_dt=dt, seed=7)
    assert res.decision is SPRTDecision.REJECT
    assert res.rejected
    assert res.n_samples < _fixed_window_samples(max_t, dt)
    assert res.test_time < max_t


def test_sprt_decisions_robust_across_seeds():
    """The accept/reject calls hold over many noise realizations (no flips)."""
    m = UncertaintyModel()
    reject_limit = sccm_to_pa_m3_s(5.0)
    C_good = conductance_from_leak_rate(sccm_to_pa_m3_s(1.0), m.test_pressure)
    C_bad = conductance_from_leak_rate(sccm_to_pa_m3_s(10.0), m.test_pressure)
    n_fixed = _fixed_window_samples(30.0, 0.1)
    for seed in range(25):
        rg = sequential_decision(reject_limit, m, C=C_good,
                                 max_test_time=30.0, sample_dt=0.1, seed=seed)
        rb = sequential_decision(reject_limit, m, C=C_bad,
                                 max_test_time=30.0, sample_dt=0.1, seed=seed)
        assert rg.decision is SPRTDecision.ACCEPT
        assert rb.decision is SPRTDecision.REJECT
        # both terminate well inside the fixed window
        assert rg.n_samples < n_fixed
        assert rb.n_samples < n_fixed


def test_sprt_average_test_time_below_fixed_window():
    """Average SPRT test time over a good/bad mix is far below the fixed window."""
    m = UncertaintyModel()
    reject_limit = sccm_to_pa_m3_s(5.0)
    C_good = conductance_from_leak_rate(sccm_to_pa_m3_s(1.0), m.test_pressure)
    C_bad = conductance_from_leak_rate(sccm_to_pa_m3_s(10.0), m.test_pressure)
    max_t = 30.0
    times = []
    for seed in range(30):
        C = C_good if seed % 2 == 0 else C_bad
        res = sequential_decision(reject_limit, m, C=C,
                                  max_test_time=max_t, sample_dt=0.1, seed=seed)
        times.append(res.test_time)
    assert np.mean(times) < max_t        # strictly faster on average
    assert np.mean(times) < 0.5 * max_t  # in fact dramatically faster


def test_sprt_with_thermal_transient_still_correct():
    """After an adequate settle, a residual transient keeps the calls correct.

    The thermal profile is referenced to the start of SETTLE; with ``settle_t``
    large enough that the residual transient has decayed, the SPRT sees a clean
    leak signal and the clearly-good / clearly-bad calls hold.
    """
    m = UncertaintyModel()
    reject_limit = sccm_to_pa_m3_s(5.0)
    thermal = ThermalTransient(T_ambient=m.T_ref, dT0=2.0, tau_thermal=8.0)
    C_good = conductance_from_leak_rate(sccm_to_pa_m3_s(1.0), m.test_pressure)
    C_bad = conductance_from_leak_rate(sccm_to_pa_m3_s(12.0), m.test_pressure)
    rg = sequential_decision(reject_limit, m, C=C_good, temp_profile=thermal,
                             settle_t=15.0, max_test_time=30.0, sample_dt=0.1, seed=3)
    rb = sequential_decision(reject_limit, m, C=C_bad, temp_profile=thermal,
                             settle_t=15.0, max_test_time=30.0, sample_dt=0.1, seed=3)
    assert rg.decision is SPRTDecision.ACCEPT
    assert rb.decision is SPRTDecision.REJECT


def test_sprt_short_settle_thermal_can_false_reject_good_part():
    """Too-short settle leaves a transient that *correctly* (per the data it
    sees) trips a good part to REJECT -- the settle-time-vs-error tradeoff that
    motivates the whole optimization. Longer settle fixes it.
    """
    m = UncertaintyModel()
    reject_limit = sccm_to_pa_m3_s(5.0)
    thermal = ThermalTransient(T_ambient=m.T_ref, dT0=2.0, tau_thermal=8.0)
    C_good = conductance_from_leak_rate(sccm_to_pa_m3_s(1.0), m.test_pressure)
    short = sequential_decision(reject_limit, m, C=C_good, temp_profile=thermal,
                                settle_t=0.0, max_test_time=30.0, sample_dt=0.1, seed=3)
    long = sequential_decision(reject_limit, m, C=C_good, temp_profile=thermal,
                               settle_t=20.0, max_test_time=30.0, sample_dt=0.1, seed=3)
    assert short.decision is SPRTDecision.REJECT      # false reject from transient
    assert long.decision is SPRTDecision.ACCEPT       # settle rescues it
