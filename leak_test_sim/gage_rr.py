"""Measurement-system analysis: Monte-Carlo Gage R&R / measurement capability.

A leak tester is a gauge, and before trusting its pass/fail calls you must know
its measurement uncertainty relative to the tolerance. This module runs repeated
*simulated* measurements (Monte-Carlo over the instrument noise/drift and any
thermal scatter) and reports the standard measurement-capability numbers:

* repeatability sigma (the measurement standard deviation)
* %GRR relative to the tolerance (study variation / tolerance)
* number of distinct categories (ndc), an AIAG MSA metric

In a classic Gage R&R "repeatability & reproducibility" splits the variation into
equipment variation (EV, repeatability) and appraiser variation (AV,
reproducibility). A leak tester is automated -- there is no human appraiser -- so
the dominant term is repeatability (EV). We model repeated readings of the same
part and treat their spread as the gauge's repeatability, then express it against
the tolerance band.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .physics import leak_rate_from_decay
from .sequence import SequenceConfig, run_sequence


@dataclass
class GageRRResult:
    n: int
    mean: float                  # mean measured leak rate (SI)
    sigma: float                 # repeatability std dev (SI)
    tolerance: float             # tolerance band used for %GRR (SI)
    grr_sigma: float             # study variation = k * sigma
    pct_grr: float               # 100 * grr_sigma / tolerance
    ndc: float                   # number of distinct categories
    samples: np.ndarray          # raw measured leak rates

    @property
    def capable(self) -> bool:
        """AIAG rule of thumb: %GRR < 10% is acceptable, <30% marginal."""
        return self.pct_grr < 10.0


def monte_carlo_leak_measurement(cfg: SequenceConfig, transducer, C: float,
                                 tolerance: float, n: int = 200,
                                 temp_profile=None, k: float = 6.0,
                                 seed: int | None = 0) -> GageRRResult:
    """Repeat a leak measurement ``n`` times and compute gauge capability.

    Each replicate re-runs the sequence with fresh instrument noise (the
    transducer is reseeded per replicate from a master RNG so the study is
    reproducible). The spread of the measured leak rates is the gauge
    repeatability.

    Parameters
    ----------
    cfg, C, temp_profile : passed to :func:`run_sequence`.
    transducer : the noisy transducer (its noise drives the scatter).
    tolerance : tolerance band for %GRR (e.g. the reject limit, or its span).
    n : number of Monte-Carlo replicates.
    k : study-variation multiplier (AIAG uses 6 -> +/-3 sigma = 99.73%).
    seed : master seed for reproducibility.
    """
    master = np.random.default_rng(seed)
    samples = np.empty(n, dtype=float)

    for i in range(n):
        transducer.reset(seed=int(master.integers(0, 2**31 - 1)))
        res = run_sequence(cfg, C=C, temp_profile=temp_profile,
                           transducer=transducer)
        samples[i] = res.leak_rate_si

    mean = float(np.mean(samples))
    sigma = float(np.std(samples, ddof=1))
    grr_sigma = k * sigma
    pct_grr = 100.0 * grr_sigma / tolerance if tolerance else float("inf")

    # ndc = 1.41 * (part variation / gauge variation). With a single part the
    # "part variation" is taken as the tolerance band as a practical proxy.
    ndc = 1.41 * (tolerance / grr_sigma) if grr_sigma > 0 else float("inf")

    return GageRRResult(
        n=n, mean=mean, sigma=sigma, tolerance=tolerance,
        grr_sigma=grr_sigma, pct_grr=pct_grr, ndc=ndc, samples=samples,
    )


def noise_to_leak_rate_sigma(noise_std: float, V: float, dt: float) -> float:
    """Analytic repeatability of the leak rate from transducer noise alone.

    ΔP = P_end - P_start of two independent noisy samples, each with std
    ``noise_std``, so ``std(ΔP) = sqrt(2) * noise_std`` and the leak-rate std is
    ``sqrt(2) * noise_std * V / dt``. Gives a closed-form check on the
    Monte-Carlo sigma (resolution/drift aside).
    """
    return leak_rate_from_decay(np.sqrt(2.0) * noise_std, V, dt)
