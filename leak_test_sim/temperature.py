"""Thermal transients and temperature compensation.

Why this matters
----------------
In pressure-decay leak testing, the *dominant* false-reject source is not real
leaks -- it is temperature. Filling a part with compressed gas does adiabatic
compression work and warms the gas; the part walls then pull that heat away and
the gas cools back toward ambient during the settle/test window. By the ideal
gas law at fixed volume and moles:

    dP/P = dT/T

So a cooling gas (dT < 0) drops pressure with *no real leak* -- indistinguishable
from a leak to an uncompensated instrument. A 1 K cool-down on a part at 300 kPa
absolute, 293 K, fakes a pressure drop of:

    dP = P * dT/T = 300000 * (-1)/293 ~= -1024 Pa

which on a small volume can dwarf the real reject leak rate.

Compensation
------------
Real instruments fight this two ways (both modelled here):

1. *Settle time* -- wait for the transient to decay (see :mod:`sequence`).
2. *Active temperature compensation* -- measure the gas/part temperature and
   correct the pressure to a reference temperature:

       P_corr(t) = P(t) * (T_ref / T(t))

   Genuine mass loss survives this correction; pure thermal drift cancels.

This module provides simple, physically-motivated temperature profiles for the
fill-heat-then-cool transient and the compensation helper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .physics import temperature_corrected_pressure


@dataclass
class ThermalTransient:
    """Exponential relaxation of gas temperature back to ambient.

    Models the post-fill cool-down: the gas starts at ``T_ambient + dT0`` (dT0>0
    for fill heating) and relaxes to ``T_ambient`` with thermal time constant
    ``tau_thermal``:

        T(t) = T_ambient + dT0 * exp(-t / tau_thermal)

    ``tau_thermal`` is set by the part's wall thermal mass / heat-transfer
    coefficient -- typically seconds to tens of seconds, which is exactly why a
    too-short settle leaves a transient that corrupts the ΔP measurement.
    """

    T_ambient: float = 293.15   # K
    dT0: float = 2.0            # K initial temperature offset (fill heating)
    tau_thermal: float = 8.0    # s thermal relaxation time constant

    def __call__(self, t):
        t = np.asarray(t, dtype=float)
        return self.T_ambient + self.dT0 * np.exp(-t / self.tau_thermal)

    def temperature(self, t):
        return self(t)


def compensate(P, T, T_ref):
    """Vectorised temperature compensation ``P_corr = P * (T_ref / T)``.

    Accepts scalars or numpy arrays.
    """
    P = np.asarray(P, dtype=float)
    T = np.asarray(T, dtype=float)
    return temperature_corrected_pressure(P, T, T_ref)


def apparent_leak_from_thermal(V: float, P: float, T: float, dT: float,
                               dt: float) -> float:
    """Apparent (false) leak rate produced purely by a temperature change.

    A temperature change ``dT`` over window ``dt`` fakes a pressure change
    ``dP = P*dT/T`` (ideal gas), which an instrument reads as a leak rate
    ``|dP|*V/dt``. Returns Pa*m^3/s. Useful for sizing how much temperature
    stability a given reject limit demands.
    """
    dP = P * dT / T
    return abs(dP) * V / dt
