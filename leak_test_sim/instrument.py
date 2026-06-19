"""Instrument realism: transducer resolution, noise, and baseline drift.

A real pressure-decay tester never sees the clean analytic pressure trace. The
transducer has:

* finite *resolution* (ADC quantisation) -- pressure is reported in discrete
  steps, so very small ΔP rounds away;
* *Gaussian noise* -- electrical + sensor noise, the main contributor to
  measurement repeatability (and therefore to the guard band);
* *baseline / zero drift* -- slow offset wander that, over a test window, adds a
  spurious slope and biases ΔP just like a thermal transient.

This module wraps an ideal pressure signal with those effects so the rest of the
pipeline (ΔP extraction, pass/fail, Gage R&R) operates on realistic data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Transducer:
    """A simulated pressure transducer.

    Parameters
    ----------
    resolution : smallest resolvable pressure step (Pa). The signal is rounded
        to a multiple of this. ``0`` disables quantisation.
    noise_std : standard deviation of additive Gaussian noise (Pa).
    drift_rate : baseline drift slope (Pa/s), applied as a linear offset over
        time. Positive = baseline rises.
    fullscale : optional full-scale range (Pa); readings are clipped to
        ``[-fullscale, fullscale]`` if given. ``None`` = no clipping.
    seed : RNG seed for reproducible noise. ``None`` = fresh entropy.
    """

    resolution: float = 1.0
    noise_std: float = 5.0
    drift_rate: float = 0.0
    fullscale: float | None = None
    seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def reset(self, seed: int | None = None):
        """Reseed the internal RNG (for deterministic repeated runs)."""
        self._rng = np.random.default_rng(self.seed if seed is None else seed)

    def quantize(self, P):
        """Round pressure to the transducer resolution."""
        if self.resolution and self.resolution > 0:
            return np.round(np.asarray(P, dtype=float) / self.resolution) * self.resolution
        return np.asarray(P, dtype=float)

    def measure(self, t, P_true):
        """Apply noise + drift + quantisation to a true pressure trace.

        Parameters
        ----------
        t : time vector (s), same shape as ``P_true`` (used for drift).
        P_true : ideal pressure trace (Pa).

        Returns the measured (noisy, drifted, quantised) pressure trace.
        """
        t = np.asarray(t, dtype=float)
        P_true = np.asarray(P_true, dtype=float)

        drift = self.drift_rate * (t - t[0]) if t.size else 0.0
        noise = self._rng.normal(0.0, self.noise_std, size=P_true.shape) if self.noise_std > 0 else 0.0

        P_meas = P_true + drift + noise
        if self.fullscale is not None:
            P_meas = np.clip(P_meas, -self.fullscale, self.fullscale)
        return self.quantize(P_meas)
