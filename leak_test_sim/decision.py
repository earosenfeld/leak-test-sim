"""Pass/Fail decision with a guard band for measurement uncertainty.

A leak test compares a measured leak rate against a *reject limit* (the largest
leak the part may have and still be good). Because the measurement itself has
uncertainty, deciding right at the limit risks passing bad parts (escapes) or
failing good ones (false rejects). A **guard band** pulls the accept threshold
in by the measurement uncertainty:

    measured >= reject_limit                  -> REJECT  (clearly too leaky)
    measured <  reject_limit - guard_band     -> ACCEPT  (clearly good)
    otherwise                                 -> INDETERMINATE (retest)

The guard band is typically sized from the gauge uncertainty -- e.g. a multiple
(k) of the measurement standard deviation (see :mod:`gage_rr`). Guarding only
the accept side is the conservative (ship-no-bad-parts) convention used on
production leak testers; this keeps escapes low at the cost of some retests.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Verdict(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    INDETERMINATE = "indeterminate"   # retest


@dataclass
class DecisionConfig:
    """Pass/fail thresholds (all leak rates in the same unit, default SI)."""
    reject_limit: float          # reject if measured >= this
    guard_band: float = 0.0      # accept only if measured < reject_limit - guard


@dataclass
class Decision:
    verdict: Verdict
    measured: float
    reject_limit: float
    guard_band: float
    accept_threshold: float      # reject_limit - guard_band
    margin: float                # reject_limit - measured (negative => over limit)

    @property
    def passed(self) -> bool:
        return self.verdict is Verdict.ACCEPT

    @property
    def failed(self) -> bool:
        return self.verdict is Verdict.REJECT


def decide(measured: float, cfg: DecisionConfig) -> Decision:
    """Apply the guard-banded accept/reject rule to a measured leak rate."""
    accept_threshold = cfg.reject_limit - cfg.guard_band

    if measured >= cfg.reject_limit:
        verdict = Verdict.REJECT
    elif measured < accept_threshold:
        verdict = Verdict.ACCEPT
    else:
        verdict = Verdict.INDETERMINATE

    return Decision(
        verdict=verdict,
        measured=measured,
        reject_limit=cfg.reject_limit,
        guard_band=cfg.guard_band,
        accept_threshold=accept_threshold,
        margin=cfg.reject_limit - measured,
    )


def guard_band_from_uncertainty(sigma: float, k: float = 2.0) -> float:
    """Guard band sized as ``k * sigma`` of the measurement uncertainty.

    ``k=2`` (~95% coverage) is a common default; production specs sometimes use
    higher k for safety-critical parts.
    """
    return k * sigma
