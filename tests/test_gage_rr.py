"""Monte-Carlo Gage R&R / measurement capability."""

import pytest

from leak_test_sim import (
    SequenceConfig, Transducer, monte_carlo_leak_measurement,
    noise_to_leak_rate_sigma, conductance_from_leak_rate, sccm_to_pa_m3_s,
)


def test_noise_to_leak_rate_sigma_closed_form():
    # std(leak) = sqrt(2)*noise * V / dt
    noise, V, dt = 5.0, 1e-4, 5.0
    sig = noise_to_leak_rate_sigma(noise, V, dt)
    import numpy as np
    assert sig == pytest.approx(np.sqrt(2.0) * noise * V / dt)


def test_monte_carlo_sigma_matches_analytic():
    """MC repeatability sigma should match the noise-driven closed form.

    With no quantisation/drift, the only scatter source is Gaussian noise on the
    two endpoint samples, so MC sigma ~ sqrt(2)*noise*V/dt.
    """
    cfg = SequenceConfig(test_pressure=300000.0, volume=1e-4,
                         settle_time=2.0, test_time=5.0, dt=0.02)
    noise = 8.0
    tx = Transducer(resolution=0.0, noise_std=noise, drift_rate=0.0, seed=0)
    Q_true = sccm_to_pa_m3_s(5.0)
    C = conductance_from_leak_rate(Q_true, cfg.test_pressure)
    tol = sccm_to_pa_m3_s(10.0)

    res = monte_carlo_leak_measurement(cfg, tx, C=C, tolerance=tol, n=400, seed=1)
    analytic = noise_to_leak_rate_sigma(noise, cfg.volume, cfg.test_time)
    # MC sigma within ~15% of closed form (interp adds a touch of averaging)
    assert res.sigma == pytest.approx(analytic, rel=0.15)
    # mean should sit near the true leak rate
    assert res.mean == pytest.approx(Q_true, rel=0.05)


def test_grr_metrics_present_and_sane():
    cfg = SequenceConfig(test_pressure=300000.0, volume=1e-4,
                         settle_time=2.0, test_time=5.0, dt=0.02)
    tx = Transducer(resolution=1.0, noise_std=5.0, seed=0)
    Q_true = sccm_to_pa_m3_s(5.0)
    C = conductance_from_leak_rate(Q_true, cfg.test_pressure)
    tol = sccm_to_pa_m3_s(10.0)
    res = monte_carlo_leak_measurement(cfg, tx, C=C, tolerance=tol, n=200, seed=2)
    assert res.pct_grr > 0.0
    assert res.ndc > 0.0
    assert res.grr_sigma == pytest.approx(6.0 * res.sigma)
    assert len(res.samples) == 200


def test_lower_noise_improves_grr():
    cfg = SequenceConfig(test_pressure=300000.0, volume=1e-4,
                         settle_time=2.0, test_time=5.0, dt=0.02)
    Q_true = sccm_to_pa_m3_s(5.0)
    C = conductance_from_leak_rate(Q_true, cfg.test_pressure)
    tol = sccm_to_pa_m3_s(10.0)

    noisy = Transducer(resolution=0.0, noise_std=20.0, seed=0)
    quiet = Transducer(resolution=0.0, noise_std=2.0, seed=0)
    r_noisy = monte_carlo_leak_measurement(cfg, noisy, C=C, tolerance=tol, n=300, seed=3)
    r_quiet = monte_carlo_leak_measurement(cfg, quiet, C=C, tolerance=tol, n=300, seed=3)
    assert r_quiet.pct_grr < r_noisy.pct_grr
    assert r_quiet.sigma < r_noisy.sigma
