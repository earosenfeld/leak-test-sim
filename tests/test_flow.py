"""Flow regimes (Knudsen), Poiseuille vs molecular, gas correlation."""

import math

import pytest

from leak_test_sim import (
    AIR, HELIUM, mean_free_path, knudsen_number, flow_regime,
    poiseuille_conductance, molecular_conductance,
    viscous_correlation, molecular_correlation, correlate_leak_rate,
)
from leak_test_sim.flow import ETA_AIR, ETA_HE, MW_AIR, MW_HE


def test_mean_free_path_atmospheric_air():
    # air at 1 atm, room temp -> ~68 nm (well-known textbook value)
    lam = mean_free_path(101325.0, 293.15, AIR)
    assert lam == pytest.approx(68e-9, rel=0.25)


def test_mean_free_path_scales_inverse_pressure():
    lam1 = mean_free_path(100000.0, 293.15, AIR)
    lam2 = mean_free_path(200000.0, 293.15, AIR)
    assert lam1 / lam2 == pytest.approx(2.0, rel=1e-12)


def test_flow_regime_classification():
    assert flow_regime(0.001) == "continuum"
    assert flow_regime(0.1) == "transitional"
    assert flow_regime(5.0) == "molecular"


def test_knudsen_large_channel_is_continuum():
    # 1 mm channel at 1 atm -> Kn tiny -> continuum
    Kn = knudsen_number(1e-3, 101325.0, 293.15, AIR)
    assert Kn < 0.01
    assert flow_regime(Kn) == "continuum"


def test_knudsen_tiny_channel_is_molecular():
    # 10 nm channel at 1 atm -> Kn > 1 -> molecular
    Kn = knudsen_number(10e-9, 101325.0, 293.15, AIR)
    assert Kn > 1.0
    assert flow_regime(Kn) == "molecular"


def test_poiseuille_scales_d4_and_inverse_eta():
    base = poiseuille_conductance(1e-5, 1e-3, 200000.0, eta=ETA_AIR)
    bigger = poiseuille_conductance(2e-5, 1e-3, 200000.0, eta=ETA_AIR)
    # d^4 -> 16x
    assert bigger / base == pytest.approx(16.0, rel=1e-9)
    # 1/eta scaling
    he = poiseuille_conductance(1e-5, 1e-3, 200000.0, eta=ETA_HE)
    assert he / base == pytest.approx(ETA_AIR / ETA_HE, rel=1e-9)


def test_molecular_scales_d3_and_inverse_sqrt_mw():
    base = molecular_conductance(1e-6, 1e-3, 293.15, mw=MW_AIR)
    bigger = molecular_conductance(2e-6, 1e-3, 293.15, mw=MW_AIR)
    assert bigger / base == pytest.approx(8.0, rel=1e-9)   # d^3
    he = molecular_conductance(1e-6, 1e-3, 293.15, mw=MW_HE)
    # C ∝ 1/sqrt(MW) -> He/air = sqrt(MW_air/MW_he)
    assert he / base == pytest.approx(math.sqrt(MW_AIR / MW_HE), rel=1e-9)


def test_viscous_correlation_air_leaks_faster_than_he():
    # viscous: Q ∝ 1/eta, eta_air < eta_he -> air leaks faster, He reads smaller
    r = viscous_correlation(AIR, HELIUM)        # Q_He / Q_air
    assert r == pytest.approx(ETA_AIR / ETA_HE, rel=1e-12)
    assert r < 1.0


def test_molecular_correlation_he_leaks_faster_than_air():
    # molecular: Q ∝ 1/sqrt(MW), He much lighter -> He leaks faster
    r = molecular_correlation(AIR, HELIUM)      # Q_He / Q_air
    assert r == pytest.approx(math.sqrt(MW_AIR / MW_HE), rel=1e-12)
    assert r > 1.0
    # ~2.69x for air->He molecular
    assert r == pytest.approx(2.69, rel=0.02)


def test_correlate_leak_rate_dispatch():
    Q_air = 1.0
    assert correlate_leak_rate(Q_air, AIR, HELIUM, "viscous") == pytest.approx(
        viscous_correlation(AIR, HELIUM))
    assert correlate_leak_rate(Q_air, AIR, HELIUM, "molecular") == pytest.approx(
        molecular_correlation(AIR, HELIUM))
    with pytest.raises(ValueError):
        correlate_leak_rate(Q_air, AIR, HELIUM, "nonsense")


def test_correlation_roundtrip():
    # air->He->air should return original (both regimes)
    for regime in ("viscous", "molecular"):
        Q = 3.3
        to_he = correlate_leak_rate(Q, AIR, HELIUM, regime)
        back = correlate_leak_rate(to_he, HELIUM, AIR, regime)
        assert back == pytest.approx(Q, rel=1e-12)
