"""Test-sequence state machine: fill -> settle -> test -> exhaust.

A pressure-decay cycle runs through four phases:

1. **FILL**     -- charge the part from atmosphere up to the test pressure.
2. **SETTLE**   -- (a.k.a. stabilize/dwell) hold isolated while fill-induced
                   pressure and *temperature* transients decay. No measurement.
3. **TEST**     -- isolated measurement window; pressure is sampled at the start
                   and end and ΔP is taken over this window only.
4. **EXHAUST**  -- vent the part back to atmosphere.

The two measurement-relevant facts the engine makes explicit:

* ΔP is taken **only over the TEST window**, after SETTLE -- never across the
  fill transient.
* If SETTLE is too short, the fill temperature transient (and any pressure
  overshoot) is still decaying *into* the TEST window, adding a spurious slope
  that biases ΔP. :func:`settle_error_tradeoff` quantifies this.

During SETTLE and TEST the part is isolated, so it follows the leak+thermal
physics from :mod:`physics`. The engine drives a single continuous ideal-gas
simulation across SETTLE+TEST (so the thermal transient carries over correctly
from settle into test) and then reports the ΔP over the TEST sub-window.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .physics import simulate_decay, leak_rate_from_decay, DecayResult
from .temperature import compensate
from .units import ATM, all_units


class Phase(str, Enum):
    FILL = "fill"
    SETTLE = "settle"
    TEST = "test"
    EXHAUST = "exhaust"


@dataclass
class SequenceConfig:
    """Configuration for a fill->settle->test->exhaust cycle."""
    test_pressure: float          # absolute fill target (Pa)
    volume: float                 # test volume V (m^3)
    fill_time: float = 3.0        # s
    settle_time: float = 10.0     # s
    test_time: float = 5.0        # s
    exhaust_time: float = 2.0     # s
    P_atm: float = ATM            # Pa
    dt: float = 0.01              # simulation time step (s)


@dataclass
class MeasurementResult:
    """Outcome of a measured test window."""
    dP: float                     # signed pressure change over TEST window (Pa)
    dt_test: float                # TEST window length (s)
    volume: float                 # V (m^3)
    leak_rate_si: float           # measured leak rate (Pa*m^3/s), magnitude
    P_start: float                # absolute pressure at start of TEST (Pa)
    P_end: float                  # absolute pressure at end of TEST (Pa)
    decay: DecayResult            # full settle+test trace (uncompensated)
    dP_compensated: float | None = None       # ΔP after temp compensation
    leak_rate_si_compensated: float | None = None

    @property
    def leak_rate_units(self) -> dict[str, float]:
        """Measured leak rate expressed in all supported units."""
        return all_units(self.leak_rate_si)

    @property
    def leak_rate_units_compensated(self) -> dict[str, float] | None:
        if self.leak_rate_si_compensated is None:
            return None
        return all_units(self.leak_rate_si_compensated)


def run_sequence(cfg: SequenceConfig, C: float = 0.0,
                 temp_profile=None, T_ref: float | None = None,
                 transducer=None) -> MeasurementResult:
    """Run a full cycle and return the ΔP/leak-rate measured over the TEST window.

    Parameters
    ----------
    cfg : the sequence configuration.
    C : leak conductance (m^3/s); 0 = perfectly sealed part.
    temp_profile : optional callable ``t -> T(K)`` describing the gas-temperature
        transient. ``t=0`` is taken as the **start of SETTLE** (i.e. just after
        fill), so the transient decays through SETTLE and into TEST exactly as on
        a real part.
    T_ref : if given, also compute the temperature-compensated ΔP/leak-rate using
        ``P_corr = P*(T_ref/T)``. Requires ``temp_profile``.
    transducer : optional :class:`~leak_test_sim.instrument.Transducer` to add
        realism (noise/drift/resolution) to the measured pressures.

    Notes
    -----
    Only SETTLE+TEST are physically simulated (the isolated phases). FILL and
    EXHAUST are bookkeeping for the timeline; no measurement happens there.
    """
    # Simulate the isolated SETTLE+TEST span as one continuous decay so the
    # thermal transient flows from settle into the test window.
    span = cfg.settle_time + cfg.test_time
    T0 = float(temp_profile(0.0)) if temp_profile is not None else 293.15

    decay = simulate_decay(
        P0=cfg.test_pressure, V=cfg.volume, C=C, duration=span,
        dt=cfg.dt, P_atm=cfg.P_atm, T0=T0, temp_profile=temp_profile,
    )

    t_test_start = cfg.settle_time
    t_test_end = cfg.settle_time + cfg.test_time

    P_true = decay.P
    t = decay.t

    # Optionally corrupt with instrument effects before reading ΔP.
    if transducer is not None:
        P_meas = transducer.measure(t, P_true)
    else:
        P_meas = P_true

    P_start = float(np.interp(t_test_start, t, P_meas))
    P_end = float(np.interp(t_test_end, t, P_meas))
    dP = P_end - P_start
    leak_rate_si = leak_rate_from_decay(dP, cfg.volume, cfg.test_time)

    result = MeasurementResult(
        dP=dP, dt_test=cfg.test_time, volume=cfg.volume,
        leak_rate_si=leak_rate_si, P_start=P_start, P_end=P_end, decay=decay,
    )

    if T_ref is not None and temp_profile is not None:
        P_corr = compensate(P_meas, decay.T, T_ref)
        Pc_start = float(np.interp(t_test_start, t, P_corr))
        Pc_end = float(np.interp(t_test_end, t, P_corr))
        dP_c = Pc_end - Pc_start
        result.dP_compensated = dP_c
        result.leak_rate_si_compensated = leak_rate_from_decay(
            dP_c, cfg.volume, cfg.test_time)

    return result


def settle_error_tradeoff(cfg: SequenceConfig, thermal, C: float = 0.0,
                          settle_times=None) -> list[dict]:
    """Quantify measurement error vs settle time for a thermal transient.

    For each candidate settle time, runs the cycle with the given thermal
    transient (no compensation) and reports the measured leak rate. With C=0 the
    *true* leak rate is zero, so the entire reported value is thermal-transient
    error -- showing how a longer settle drives the false reading toward zero.

    Returns a list of dicts: ``{settle_time, leak_rate_si, dP, dT_over_window}``.
    """
    if settle_times is None:
        settle_times = [1.0, 2.0, 5.0, 10.0, 20.0, 40.0]

    rows = []
    for st in settle_times:
        c = SequenceConfig(
            test_pressure=cfg.test_pressure, volume=cfg.volume,
            fill_time=cfg.fill_time, settle_time=st, test_time=cfg.test_time,
            exhaust_time=cfg.exhaust_time, P_atm=cfg.P_atm, dt=cfg.dt,
        )
        res = run_sequence(c, C=C, temp_profile=thermal)
        T_at_start = float(thermal(st))
        T_at_end = float(thermal(st + cfg.test_time))
        rows.append({
            "settle_time": st,
            "leak_rate_si": res.leak_rate_si,
            "dP": res.dP,
            "dT_over_window": T_at_end - T_at_start,
        })
    return rows
