"""Instrument realism: resolution, noise, drift."""

import numpy as np
import pytest

from leak_test_sim import Transducer


def test_quantization():
    t = Transducer(resolution=10.0, noise_std=0.0, drift_rate=0.0, seed=0)
    P = np.array([100003.0, 100007.0, 99996.0])
    q = t.quantize(P)
    assert np.allclose(q, [100000.0, 100010.0, 100000.0])


def test_zero_resolution_no_quantization():
    t = Transducer(resolution=0.0, noise_std=0.0, seed=0)
    P = np.array([1.234, 5.678])
    assert np.allclose(t.quantize(P), P)


def test_noise_is_reproducible_with_seed():
    t1 = Transducer(resolution=0.0, noise_std=5.0, seed=42)
    t2 = Transducer(resolution=0.0, noise_std=5.0, seed=42)
    tv = np.linspace(0, 5, 100)
    P = np.full_like(tv, 100000.0)
    assert np.allclose(t1.measure(tv, P), t2.measure(tv, P))


def test_noise_statistics():
    t = Transducer(resolution=0.0, noise_std=7.0, drift_rate=0.0, seed=1)
    tv = np.linspace(0, 10, 50000)
    P = np.full_like(tv, 100000.0)
    meas = t.measure(tv, P)
    resid = meas - 100000.0
    assert np.std(resid) == pytest.approx(7.0, rel=0.02)
    assert np.mean(resid) == pytest.approx(0.0, abs=0.1)


def test_drift_is_linear():
    t = Transducer(resolution=0.0, noise_std=0.0, drift_rate=2.0, seed=0)
    tv = np.array([0.0, 1.0, 5.0, 10.0])
    P = np.full_like(tv, 1000.0)
    meas = t.measure(tv, P)
    # offset = drift_rate * (t - t0)
    assert np.allclose(meas, 1000.0 + 2.0 * tv)


def test_fullscale_clipping():
    t = Transducer(resolution=0.0, noise_std=0.0, fullscale=500.0, seed=0)
    tv = np.array([0.0, 1.0])
    P = np.array([600.0, -600.0])
    meas = t.measure(tv, P)
    assert np.allclose(meas, [500.0, -500.0])


def test_reset_restores_sequence():
    t = Transducer(resolution=0.0, noise_std=3.0, seed=7)
    tv = np.linspace(0, 1, 10)
    P = np.zeros_like(tv)
    a = t.measure(tv, P)
    t.reset()
    b = t.measure(tv, P)
    assert np.allclose(a, b)
