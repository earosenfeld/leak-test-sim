"""Flow-regime physics: laminar (Poiseuille) vs molecular flow, and gas correlation.

Leak paths behave differently depending on how the mean free path of the gas
compares to the channel size -- the **Knudsen number** Kn:

    Kn = lambda / d        (lambda = mean free path, d = characteristic diameter)

    Kn < 0.01      continuum / viscous (laminar Poiseuille) flow
    0.01 < Kn < 1  transitional / slip
    Kn > 1         (free) molecular flow

This matters for leak testing because the **gas correlation factor** -- how a
leak measured with one gas (e.g. air on the line) relates to the spec gas (e.g.
helium on a tracer-gas station) -- depends on the regime:

* Viscous/laminar: conductance ∝ 1/eta (dynamic viscosity). Air leaks *faster*
  than helium through the same viscous path because eta_air < eta_He.
* Molecular: conductance ∝ 1/sqrt(MW). Helium leaks *faster* than air because
  it is far lighter (MW_He << MW_air).

Constants used (typical, ~293 K):
    eta_air = 1.81e-5 Pa.s      MW_air = 28.97 g/mol
    eta_He  = 1.96e-5 Pa.s      MW_He  = 4.00  g/mol
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --- gas properties (~293 K) --------------------------------------------------

ETA_AIR = 1.81e-5   # Pa.s  dynamic viscosity of air
ETA_HE = 1.96e-5    # Pa.s  dynamic viscosity of helium
MW_AIR = 28.97      # g/mol molar mass of air
MW_HE = 4.00        # g/mol molar mass of helium

# Boltzmann constant
K_B = 1.380649e-23  # J/K


@dataclass(frozen=True)
class Gas:
    name: str
    eta: float          # dynamic viscosity (Pa.s)
    mw: float           # molar mass (g/mol)
    d_molecule: float   # effective molecular collision diameter (m)


AIR = Gas("air", ETA_AIR, MW_AIR, 3.7e-10)
HELIUM = Gas("helium", ETA_HE, MW_HE, 2.6e-10)


# --- Knudsen / mean free path -------------------------------------------------

def mean_free_path(P: float, T: float = 293.15, gas: Gas = AIR) -> float:
    """Mean free path (m) of a gas at pressure P (Pa), temperature T (K).

    Kinetic theory: ``lambda = k_B T / (sqrt(2) * pi * d^2 * P)``.
    """
    return K_B * T / (math.sqrt(2.0) * math.pi * gas.d_molecule ** 2 * P)


def knudsen_number(d_channel: float, P: float, T: float = 293.15,
                   gas: Gas = AIR) -> float:
    """Knudsen number ``Kn = lambda / d_channel`` for a leak channel diameter."""
    return mean_free_path(P, T, gas) / d_channel


def flow_regime(Kn: float) -> str:
    """Classify a Knudsen number into a flow regime label."""
    if Kn < 0.01:
        return "continuum"      # viscous / laminar Poiseuille
    if Kn < 1.0:
        return "transitional"   # slip / transition
    return "molecular"          # free molecular


# --- conductances of a cylindrical channel -----------------------------------

def poiseuille_conductance(d: float, length: float, P_mean: float,
                           eta: float = ETA_AIR) -> float:
    """Laminar (Poiseuille) volumetric conductance of a round tube (m^3/s).

    For viscous flow of a compressible gas through a long round channel of
    diameter ``d`` and length ``length`` at mean pressure ``P_mean``:

        C = (pi * d^4 / (128 * eta * length)) * P_mean

    The ``P_mean`` factor makes this a *throughput* conductance (Pa*m^3/s per Pa
    of driving pressure = m^3/s), and it is the term that makes viscous
    conductance ∝ 1/eta.
    """
    return (math.pi * d ** 4) / (128.0 * eta * length) * P_mean


def molecular_conductance(d: float, length: float, T: float = 293.15,
                          mw: float = MW_AIR) -> float:
    """Free-molecular volumetric conductance of a round tube (m^3/s).

    Long-tube molecular conductance (Knudsen):

        C = (pi/12) * (d^3 / length) * v_mean,   v_mean = sqrt(8 R T / (pi M))

    where ``M`` is molar mass in kg/mol. This is the term that makes molecular
    conductance ∝ 1/sqrt(MW).
    """
    R = 8.314462618  # J/(mol K)
    M = mw * 1e-3    # kg/mol
    v_mean = math.sqrt(8.0 * R * T / (math.pi * M))
    return (math.pi / 12.0) * (d ** 3 / length) * v_mean


# --- gas correlation factors --------------------------------------------------

def viscous_correlation(gas_from: Gas, gas_to: Gas) -> float:
    """Leak-rate ratio Q_to/Q_from for *viscous* (laminar) flow.

    Viscous conductance ∝ 1/eta, so ``Q_to/Q_from = eta_from / eta_to`` (same
    pressures). E.g. air->helium: air leaks faster, so helium reads smaller.
    """
    return gas_from.eta / gas_to.eta


def molecular_correlation(gas_from: Gas, gas_to: Gas) -> float:
    """Leak-rate ratio Q_to/Q_from for *molecular* flow.

    Molecular conductance ∝ 1/sqrt(MW), so
    ``Q_to/Q_from = sqrt(MW_from / MW_to)``. E.g. air->helium: helium is lighter,
    so it leaks faster (ratio > 1).
    """
    return math.sqrt(gas_from.mw / gas_to.mw)


def correlate_leak_rate(Q_from: float, gas_from: Gas, gas_to: Gas,
                        regime: str = "molecular") -> float:
    """Convert a leak rate measured with one gas to the equivalent for another.

    ``regime`` is ``"viscous"``/``"continuum"`` or ``"molecular"``. Returns the
    leak rate the *target* gas would exhibit through the same physical leak.
    """
    if regime in ("viscous", "laminar", "continuum"):
        return Q_from * viscous_correlation(gas_from, gas_to)
    if regime == "molecular":
        return Q_from * molecular_correlation(gas_from, gas_to)
    raise ValueError(f"unknown regime {regime!r}")
