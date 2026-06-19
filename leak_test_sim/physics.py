"""Pressure-decay leak-test physics.

Model
-----
A sealed test volume ``V`` (m^3) holds gas at absolute pressure ``P`` (Pa). It
leaks to atmosphere ``P_atm`` through a leak path with *leak conductance*
``C`` (m^3/s). For an isothermal, fixed-volume part the molar balance from the
ideal gas law (``PV = nRT``) gives:

    dP/dt = -(C / V) * (P - P_atm)

i.e. the *gauge* pressure ``(P - P_atm)`` decays exponentially with time
constant:

    tau = V / C

Closed-form solution:

    P(t) - P_atm = (P0 - P_atm) * exp(-t / tau)

Leak rate (throughput) at any instant, in Pa*m^3/s:

    Q(t) = C * (P(t) - P_atm)            # the instantaneous leak throughput
         = -V * dP/dt                    # equivalently, gas leaving the volume

Pressure-decay instruments do not measure ``Q`` directly. They fill the part,
let it settle, then watch pressure fall by ``dP`` over a fixed test window
``dt`` and infer:

    Q_measured ~= |dP| * V / dt          # magnitude, valid when dt << tau

This is exact in the limit ``dt -> 0`` and a small under-estimate for finite
``dt`` because the decay is exponential (the instrument sees the *average*
slope over the window, which is slightly below the initial slope). The helper
:func:`leak_rate_from_decay` returns this measured value, and
:func:`leak_rate_exact_window` returns the exponential-corrected value so the
two can be compared.

Sign convention: ``Q`` and the reported leak rate are positive for a part that
*loses* pressure to a lower-pressure environment (the normal pressure-decay
case). ``dP = P_end - P_start`` is therefore negative; we report magnitudes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .units import ATM


# -----------------------------------------------------------------------------
# Conductance <-> leak rate relationships
# -----------------------------------------------------------------------------

def conductance_from_leak_rate(Q: float, P: float, P_atm: float = ATM) -> float:
    """Leak conductance C (m^3/s) from a leak rate Q (Pa*m^3/s) at pressure P.

    From ``Q = C * (P - P_atm)``  ->  ``C = Q / (P - P_atm)``.
    """
    dP = P - P_atm
    if dP == 0:
        raise ValueError("P must differ from P_atm to define a conductance")
    return Q / dP


def leak_rate_from_conductance(C: float, P: float, P_atm: float = ATM) -> float:
    """Leak rate Q (Pa*m^3/s) from conductance C at absolute pressure P.

    ``Q = C * (P - P_atm)``.
    """
    return C * (P - P_atm)


def time_constant(V: float, C: float) -> float:
    """Decay time constant tau = V / C (seconds)."""
    if C <= 0:
        raise ValueError("conductance C must be positive")
    return V / C


# -----------------------------------------------------------------------------
# Closed-form pressure decay
# -----------------------------------------------------------------------------

def pressure_decay(t, P0: float, V: float, C: float, P_atm: float = ATM):
    """Absolute pressure ``P(t)`` for exponential decay to ``P_atm``.

    ``P(t) = P_atm + (P0 - P_atm) * exp(-t / tau)``,  ``tau = V / C``.

    ``t`` may be a scalar or a numpy array (returns the same shape).
    """
    tau = time_constant(V, C)
    t = np.asarray(t, dtype=float)
    return P_atm + (P0 - P_atm) * np.exp(-t / tau)


def delta_p_over_window(P0: float, V: float, C: float, dt: float,
                        P_atm: float = ATM) -> float:
    """Signed pressure change ``dP = P(dt) - P(0)`` over a test window dt.

    Negative for a leaking part (pressure falls). Uses the exact exponential.
    """
    P_end = float(pressure_decay(dt, P0, V, C, P_atm))
    return P_end - P0


def leak_rate_from_decay(dP: float, V: float, dt: float) -> float:
    """Measured leak rate from an observed pressure drop.

    ``Q_measured = |dP| * V / dt``  (Pa*m^3/s).

    This is the formula an instrument applies. It is the magnitude of the
    *average* leak throughput over the window.
    """
    return abs(dP) * V / dt


def leak_rate_exact_window(P0: float, V: float, C: float, dt: float,
                           P_atm: float = ATM) -> float:
    """Exact average leak throughput over a finite window (Pa*m^3/s).

    The number of moles lost over ``dt`` corresponds to a pressure change
    ``dP``; the average throughput is ``|dP| * V / dt``. Because the decay is
    exponential this is slightly below the *initial* instantaneous
    ``Q0 = C*(P0 - P_atm)``. Returned as a positive magnitude.
    """
    dP = delta_p_over_window(P0, V, C, dt, P_atm)
    return leak_rate_from_decay(dP, V, dt)


# -----------------------------------------------------------------------------
# Ideal-gas thermal effect (basis for temperature compensation)
# -----------------------------------------------------------------------------

def apparent_dp_from_dt(P: float, T: float, dT: float) -> float:
    """Apparent pressure change from a temperature change at fixed V, n.

    Ideal gas at fixed volume & moles: ``P/T = const`` -> ``dP/P = dT/T``, so

        dP = P * dT / T

    A part that warms (dT>0) shows rising pressure (masking a leak); a part that
    cools (dT<0) shows falling pressure that *looks like* a leak even with no
    real leak. This is the dominant false-reject mechanism in pressure decay.
    """
    return P * dT / T


def temperature_corrected_pressure(P: float, T: float, T_ref: float) -> float:
    """Correct a measured pressure to a reference temperature (fixed V, n).

    ``P_corr = P * (T_ref / T)``. Removes the ideal-gas thermal component so
    that only genuine mass loss (a real leak) remains in the corrected trace.
    """
    return P * (T_ref / T)


# -----------------------------------------------------------------------------
# Full ODE simulation (leak + optional time-varying temperature)
# -----------------------------------------------------------------------------

@dataclass
class DecayResult:
    """Result of a pressure-decay simulation."""
    t: np.ndarray            # time vector (s)
    P: np.ndarray            # absolute pressure trace (Pa)
    T: np.ndarray            # gas temperature trace (K)
    P_atm: float             # atmosphere used (Pa)
    V: float                 # test volume (m^3)
    C: float                 # leak conductance (m^3/s)

    @property
    def tau(self) -> float:
        return time_constant(self.V, self.C)

    def dp(self, t_start: float, t_end: float) -> float:
        """Signed pressure change between two times (interpolated)."""
        p_start = float(np.interp(t_start, self.t, self.P))
        p_end = float(np.interp(t_end, self.t, self.P))
        return p_end - p_start


def simulate_decay(P0: float, V: float, C: float, duration: float,
                   dt: float = 0.01, P_atm: float = ATM,
                   T0: float = 293.15, temp_profile=None) -> DecayResult:
    """Simulate pressure decay, optionally with a time-varying gas temperature.

    Integrates the coupled relation. With both a leak and a temperature
    transient ``T(t)`` the fixed-volume ideal-gas balance is:

        P(t) = (n(t) R T(t)) / V

    where moles ``n`` decrease through the leak. We integrate the moles lost
    (driven by the gauge pressure) and recompute ``P`` from the current ``T``
    each step, so thermal expansion and mass loss combine exactly as on a real
    part.

    Parameters
    ----------
    P0 : initial absolute pressure (Pa)
    V : test volume (m^3)
    C : leak conductance (m^3/s); use 0 for a perfectly sealed (no-leak) part
    duration : simulation length (s)
    dt : time step (s)
    P_atm : atmospheric pressure (Pa)
    T0 : initial gas temperature (K)
    temp_profile : optional callable t -> T(K). If given it overrides the
        isothermal assumption and drives the thermal term.
    """
    n_steps = int(round(duration / dt)) + 1
    t = np.linspace(0.0, duration, n_steps)

    if temp_profile is None:
        T = np.full_like(t, T0)
    else:
        T = np.asarray([float(temp_profile(ti)) for ti in t], dtype=float)

    # n*R is constant scaling; track the quantity (n*R) so P = (nR) * T / V.
    # Initial: P0 = (nR)0 * T0 / V  ->  (nR)0 = P0 * V / T0
    nR = P0 * V / T0  # this is n0 * R

    P = np.empty_like(t)
    P[0] = P0
    nR_current = nR

    for i in range(1, len(t)):
        # leak throughput uses the pressure at the *start* of the step
        P_prev = P[i - 1]
        # molar loss: dn/dt * R = -(C/V) * (P - P_atm) * V / T ... but simpler to
        # work in (nR) directly. Q = C*(P - P_atm) [Pa*m^3/s] is volumetric*P.
        # Moles lost: dn = Q*dt / (R T). Multiply by R: d(nR) = Q*dt / T.
        Q = C * (P_prev - P_atm)
        d_nR = -(Q * dt) / T[i - 1]
        nR_current = nR_current + d_nR
        P[i] = nR_current * T[i] / V

    return DecayResult(t=t, P=P, T=T, P_atm=P_atm, V=V, C=C)
