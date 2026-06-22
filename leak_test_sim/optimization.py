"""Cycle-time optimization: the settle-time-vs-error tradeoff, made quantitative.

On a production line, cycle time is money: every second of SETTLE + TEST is
throughput the station does not have. But cutting those times degrades the
measurement two different ways, and this module models both so a minimum cycle
that still meets a target uncertainty can be *computed* rather than guessed.

The uncertainty model
---------------------
The measured leak rate ``Q = |dP| * V / test`` carries two independent error
contributions:

1. **Systematic (settle-limited) error.** If SETTLE is too short the post-fill
   thermal transient is still decaying *into* the TEST window, adding a spurious
   slope to dP (the dominant false-reject mechanism -- see :mod:`temperature`).
   The residual gas-temperature offset at the start of the test window is

       dT_settle = dT0 * exp(-settle / tau_thermal)

   and over a test window of length ``test`` the gas cools by a further

       dT_window = dT_settle * (1 - exp(-test / tau_thermal))

   which the ideal-gas law turns into an apparent pressure change
   ``dP_sys = P * dT_window / T`` (see :func:`physics.apparent_dp_from_dt`) and
   hence a systematic leak-rate error

       e_sys(settle, test) = |P * dT_window / T| * V / test          (Pa*m^3/s)

   This is the *same* physics the full sequence simulator integrates; here it is
   the closed form. It falls ``~exp(-settle / tau_thermal)`` as settle grows --
   longer settle lets the transient decay -> smaller systematic error.

2. **Random (test-limited) noise error.** dP is the difference of two noisy
   transducer samples, ``std(dP) = sqrt(2) * sigma_P``, so the leak-rate
   repeatability is

       e_noise(test) = sqrt(2) * sigma_P * V / test                  (Pa*m^3/s)

   exactly :func:`gage_rr.noise_to_leak_rate_sigma`. A longer TEST grows the real
   ``|dP|`` signal against this fixed noise floor -> smaller random error
   (``~1 / test``). An optional ``noise_floor`` term models an irreducible
   uncertainty (drift, resolution, temperature-sensor accuracy) the test time
   cannot beat, so the curve flattens to a floor instead of reaching zero.

The total reported uncertainty is the root-sum-square (independent terms):

    u(settle, test) = sqrt( e_sys^2 + e_noise^2 + noise_floor^2 )

Optimization
------------
:func:`optimize_cycle_time` searches ``(settle, test)`` for the **minimum total
cycle** ``settle + test`` whose modelled ``u`` meets a target, returning the
recommended times and the achieved uncertainty. Because ``e_sys`` shrinks with
settle while ``e_noise`` shrinks with test, the meeting point is a genuine
two-axis tradeoff, not a single knob.

Sequential testing (SPRT)
-------------------------
:func:`sequential_decision` implements a Wald Sequential Probability Ratio Test
that watches the leak signal accumulate and stops as soon as the part is clearly
in- or out-of-spec -- accepting a good part or rejecting a bad one *early*,
before the full fixed TEST window elapses. On a line where most parts are clearly
good this cuts the **average** test time well below the fixed window while holding
the specified error rates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .units import ATM
from .physics import apparent_dp_from_dt
from .gage_rr import noise_to_leak_rate_sigma


# -----------------------------------------------------------------------------
# Uncertainty model
# -----------------------------------------------------------------------------

@dataclass
class UncertaintyModel:
    """Parameters of the measurement-uncertainty model.

    All pressures in Pa, volume in m^3, times in s. Defaults describe a 100 cc
    part at 3 bar absolute with a modest fill-heating transient and a typical
    transducer.
    """

    test_pressure: float = 300_000.0   # absolute test pressure P (Pa)
    volume: float = 1.0e-4             # test volume V (m^3)
    T_ref: float = 293.15              # reference gas temperature (K)
    dT0: float = 4.0                   # initial fill-heating offset (K)
    tau_thermal: float = 8.0           # thermal relaxation time constant (s)
    sigma_P: float = 5.0               # transducer noise std per sample (Pa)
    noise_floor: float = 0.0           # irreducible leak-rate uncertainty (Pa*m^3/s)
    P_atm: float = ATM                 # atmospheric pressure (Pa)


def systematic_error(settle_t: float, test_t: float,
                     model: UncertaintyModel | None = None,
                     **kw) -> float:
    """Settle-limited *systematic* leak-rate error (Pa*m^3/s), magnitude.

    Residual thermal transient at the start of the test window:
    ``dT_settle = dT0 * exp(-settle/tau)``; the gas cools a further
    ``dT_window = dT_settle * (1 - exp(-test/tau))`` over the window, which the
    ideal-gas law reads as ``dP = P*dT_window/T`` and the instrument turns into
    ``|dP|*V/test``. Decays ``~exp(-settle/tau_thermal)`` -- longer settle, less
    systematic error.
    """
    m = _resolve(model, kw)
    if test_t <= 0:
        raise ValueError("test_t must be positive")
    dT_settle = m.dT0 * np.exp(-settle_t / m.tau_thermal)
    dT_window = dT_settle * (1.0 - np.exp(-test_t / m.tau_thermal))
    dP_sys = apparent_dp_from_dt(m.test_pressure, m.T_ref, dT_window)
    return abs(dP_sys) * m.volume / test_t


def noise_error(test_t: float, model: UncertaintyModel | None = None,
                **kw) -> float:
    """Test-limited *random* leak-rate error (Pa*m^3/s), magnitude.

    ``sqrt(2) * sigma_P * V / test`` -- exactly the gauge repeatability from
    transducer noise (:func:`gage_rr.noise_to_leak_rate_sigma`). Falls ``~1/test``
    as the longer window grows the dP signal against the fixed noise.
    """
    m = _resolve(model, kw)
    if test_t <= 0:
        raise ValueError("test_t must be positive")
    return noise_to_leak_rate_sigma(m.sigma_P, m.volume, test_t)


def measurement_uncertainty(settle_t: float, test_t: float,
                            model: UncertaintyModel | None = None,
                            **kw) -> float:
    """Total measured-leak-rate uncertainty (Pa*m^3/s) vs settle & test time.

    Root-sum-square of the systematic (settle-limited) and random
    (test-limited) terms plus any irreducible ``noise_floor``:

        u = sqrt( e_sys(settle, test)^2 + e_noise(test)^2 + noise_floor^2 )

    Monotone decreasing in ``settle_t`` (systematic term decays) and in
    ``test_t`` (noise term decays), toward the ``noise_floor``.

    Parameters
    ----------
    settle_t, test_t : SETTLE and TEST window lengths (s).
    model : an :class:`UncertaintyModel`; or pass its fields as keywords.
    """
    m = _resolve(model, kw)
    e_sys = systematic_error(settle_t, test_t, m)
    e_noise = noise_error(test_t, m)
    return float(np.sqrt(e_sys ** 2 + e_noise ** 2 + m.noise_floor ** 2))


def _resolve(model: UncertaintyModel | None, kw: dict) -> UncertaintyModel:
    """Return the model to use: an explicit one, else build from keywords."""
    if model is not None:
        if kw:
            raise TypeError("pass either a model or keyword fields, not both")
        return model
    return UncertaintyModel(**kw)


# -----------------------------------------------------------------------------
# Cycle-time optimization
# -----------------------------------------------------------------------------

@dataclass
class CycleRecommendation:
    """Recommended minimum cycle that meets a target uncertainty."""
    settle_t: float                 # recommended SETTLE time (s)
    test_t: float                   # recommended TEST time (s)
    total_t: float                  # settle_t + test_t (s)
    uncertainty: float              # achieved total uncertainty (Pa*m^3/s)
    target: float                   # the target it met (Pa*m^3/s)
    systematic: float               # systematic component at the optimum (Pa*m^3/s)
    noise: float                    # noise component at the optimum (Pa*m^3/s)
    feasible: bool                  # whether the target was achievable in range

    @property
    def met(self) -> bool:
        return self.feasible and self.uncertainty <= self.target


def optimize_cycle_time(target_uncertainty: float,
                        model: UncertaintyModel | None = None,
                        settle_bounds: tuple[float, float] = (0.0, 60.0),
                        test_bounds: tuple[float, float] = (0.5, 60.0),
                        grid: int = 241,
                        **kw) -> CycleRecommendation:
    """Find the minimum-cycle ``(settle, test)`` meeting a target uncertainty.

    Scans a grid of ``(settle, test)`` over the given bounds, keeps every pair
    whose modelled :func:`measurement_uncertainty` is ``<= target_uncertainty``,
    and returns the one with the smallest **total cycle** ``settle + test`` (ties
    broken toward the lower uncertainty). The total cycle is the production cost,
    so minimizing it -- not settle or test alone -- is the right objective.

    Returns a :class:`CycleRecommendation`. If no pair in range meets the target
    (e.g. target below the noise floor), ``feasible`` is ``False`` and the
    closest-achievable (minimum-uncertainty) point is returned instead.

    Parameters
    ----------
    target_uncertainty : required total uncertainty (Pa*m^3/s).
    model : an :class:`UncertaintyModel`; or pass its fields as keywords.
    settle_bounds, test_bounds : (min, max) search range for each time (s).
    grid : number of grid points per axis.
    """
    m = _resolve(model, kw)
    if target_uncertainty <= 0:
        raise ValueError("target_uncertainty must be positive")

    settles = np.linspace(settle_bounds[0], settle_bounds[1], grid)
    tests = np.linspace(test_bounds[0], test_bounds[1], grid)

    # Vectorised uncertainty over the grid: U[i, j] for settle i, test j.
    S, T = np.meshgrid(settles, tests, indexing="ij")
    dT_settle = m.dT0 * np.exp(-S / m.tau_thermal)
    dT_window = dT_settle * (1.0 - np.exp(-T / m.tau_thermal))
    dP_sys = m.test_pressure * dT_window / m.T_ref
    e_sys = np.abs(dP_sys) * m.volume / T
    e_noise = np.sqrt(2.0) * m.sigma_P * m.volume / T
    U = np.sqrt(e_sys ** 2 + e_noise ** 2 + m.noise_floor ** 2)

    total = S + T
    feas = U <= target_uncertainty

    if np.any(feas):
        # minimum total cycle among feasible pairs; break ties on uncertainty
        cost = np.where(feas, total, np.inf)
        best = np.argmin(cost + 1e-9 * U)
        feasible = True
    else:
        # target unreachable in range -> report the lowest-uncertainty point
        best = int(np.argmin(U))
        feasible = False

    i, j = np.unravel_index(best, U.shape)
    s_opt, t_opt = float(S[i, j]), float(T[i, j])
    return CycleRecommendation(
        settle_t=s_opt, test_t=t_opt, total_t=s_opt + t_opt,
        uncertainty=float(U[i, j]), target=target_uncertainty,
        systematic=float(e_sys[i, j]), noise=float(e_noise[i, j]),
        feasible=feasible,
    )


# -----------------------------------------------------------------------------
# Sequential decision -- Wald SPRT for early accept/reject
# -----------------------------------------------------------------------------

class SPRTDecision(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    CONTINUE = "continue"   # inconclusive within the budget


@dataclass
class SPRTResult:
    """Outcome of a sequential (SPRT) leak test."""
    decision: SPRTDecision
    n_samples: int                  # samples taken before stopping
    test_time: float                # elapsed test time at the decision (s)
    leak_estimate: float            # running leak-rate estimate (Pa*m^3/s)
    log_lr: float                   # final log-likelihood ratio
    samples: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def accepted(self) -> bool:
        return self.decision is SPRTDecision.ACCEPT

    @property
    def rejected(self) -> bool:
        return self.decision is SPRTDecision.REJECT


def _wald_bounds(alpha: float, beta: float) -> tuple[float, float]:
    """Wald's log-likelihood-ratio stopping bounds (A lower, B upper).

    Accept (good) when logLR <= A = log(beta / (1 - alpha));
    reject (bad)  when logLR >= B = log((1 - beta) / alpha).
    """
    A = np.log(beta / (1.0 - alpha))
    B = np.log((1.0 - beta) / alpha)
    return A, B


def sequential_decision(reject_limit: float,
                        model: UncertaintyModel | None = None,
                        C: float = 0.0,
                        accept_margin: float = 0.5,
                        reject_margin: float = 1.5,
                        sample_dt: float = 0.1,
                        max_test_time: float = 30.0,
                        alpha: float = 0.01,
                        beta: float = 0.01,
                        temp_profile=None,
                        settle_t: float = 0.0,
                        seed: int | None = None,
                        **kw) -> SPRTResult:
    """Wald SPRT leak test: accept/reject as soon as the part is clearly in/out.

    The instrument samples the (noisy) gauge pressure every ``sample_dt`` and
    tracks the cumulative pressure drop, which is an estimate of the leak. We run
    a Sequential Probability Ratio Test between two simple hypotheses on the
    *true* leak rate:

        H0 (good): leak = accept_margin * reject_limit       (clearly in spec)
        H1 (bad) : leak = reject_margin * reject_limit       (clearly out of spec)

    Each new sample of the running leak estimate ``q_hat`` updates the
    log-likelihood ratio under Gaussian measurement noise; when it crosses
    Wald's lower bound the part is **accepted early**, when it crosses the upper
    bound it is **rejected early**. Most clearly-good (and clearly-bad) parts
    decide well before ``max_test_time`` -- cutting the *average* test time below
    a fixed window while holding the type-I/type-II error rates ``alpha``/``beta``.

    Parameters
    ----------
    reject_limit : the spec leak rate (Pa*m^3/s).
    model : :class:`UncertaintyModel` for V, P, sigma_P, thermal; or keywords.
    C : true leak conductance of the part under test (m^3/s). 0 = sealed.
    accept_margin, reject_margin : H0/H1 leak rates as multiples of the limit.
    sample_dt : interval between pressure samples (s).
    max_test_time : give up (CONTINUE) after this much test time (s).
    alpha, beta : target type-I / type-II error rates.
    temp_profile : optional callable ``t -> T(K)`` thermal transient (adds a
        systematic slope, exactly as on a real part). ``t=0`` of the profile is
        the *start of SETTLE*, matching :func:`sequence.run_sequence`.
    settle_t : SETTLE time already elapsed before the test window opens (s). The
        thermal profile is evaluated from ``settle_t`` onward, so the SPRT sees
        only the *residual* transient that survives settle -- not the full fill
        transient. Larger ``settle_t`` -> smaller systematic slope.
    seed : RNG seed for the simulated noise.
    """
    m = _resolve(model, kw)
    rng = np.random.default_rng(seed)

    q0 = accept_margin * reject_limit          # good-hypothesis leak rate
    q1 = reject_margin * reject_limit          # bad-hypothesis leak rate
    A, B = _wald_bounds(alpha, beta)

    V, P, sigma_P = m.volume, m.test_pressure, m.sigma_P
    P_atm = m.P_atm

    # True instantaneous leak throughput from the part's conductance.
    Q_true = C * (P - P_atm)                    # Pa*m^3/s, the real signal slope

    n_max = int(round(max_test_time / sample_dt))
    samples = np.empty(n_max, dtype=float)

    log_lr = 0.0
    P_start = None
    decision = SPRTDecision.CONTINUE
    n_taken = 0
    q_hat = 0.0

    for k in range(1, n_max + 1):
        t = k * sample_dt
        # ideal gauge pressure: linear drop from leak + optional thermal slope
        gauge = (P - P_atm) - Q_true * t / V
        if temp_profile is not None:
            # Thermal slope across the test window, using only the *residual*
            # transient that survives SETTLE: the profile is referenced to
            # settle_t (test-window start), so larger settle -> smaller slope.
            T_t = float(temp_profile(settle_t + t))
            T_0 = float(temp_profile(settle_t))
            gauge += P * (T_t - T_0) / m.T_ref
        P_t = P_atm + gauge + rng.normal(0.0, sigma_P)
        if P_start is None:
            P_start = P_t
        samples[k - 1] = P_t
        n_taken = k

        # running leak estimate from cumulative drop over elapsed test time
        dP = P_t - P_start
        q_hat = abs(dP) * V / t

        # Gaussian log-likelihood-ratio increment for this leak estimate. The
        # estimate's std at time t is sqrt(2)*sigma_P*V/t (two-sample dP noise).
        sd = np.sqrt(2.0) * sigma_P * V / t
        if sd <= 0:
            continue
        # logLR for H1 (bad, mean q1) vs H0 (good, mean q0)
        log_lr = ((q_hat - q0) ** 2 - (q_hat - q1) ** 2) / (2.0 * sd ** 2)

        if log_lr <= A:
            decision = SPRTDecision.ACCEPT
            break
        if log_lr >= B:
            decision = SPRTDecision.REJECT
            break

    return SPRTResult(
        decision=decision, n_samples=n_taken, test_time=n_taken * sample_dt,
        leak_estimate=q_hat, log_lr=float(log_lr),
        samples=samples[:n_taken].copy(),
    )
