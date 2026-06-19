"""leak_test_sim -- a physically-correct pressure-decay & flow leak-test simulator.

Models the full pressure-decay leak test the way production instruments (CTS,
ATEQ, Zaxis) run it: fill -> settle -> test -> exhaust, with ideal-gas pressure
decay, the dominant temperature-transient false-reject mechanism (and its
compensation), instrument noise/resolution/drift, guard-banded pass/fail, and
stretch physics for flow regimes, gas correlation, and Gage R&R.

Quick start
-----------
>>> from leak_test_sim import SequenceConfig, run_sequence, DecisionConfig, decide
>>> cfg = SequenceConfig(test_pressure=300000.0, volume=1e-4)
>>> # conductance for a 5 sccm leak at test pressure:
>>> from leak_test_sim import sccm_to_pa_m3_s, conductance_from_leak_rate
>>> Q = sccm_to_pa_m3_s(5.0)
>>> C = conductance_from_leak_rate(Q, cfg.test_pressure)
>>> res = run_sequence(cfg, C=C)
>>> d = decide(res.leak_rate_si, DecisionConfig(reject_limit=sccm_to_pa_m3_s(10.0)))
>>> d.passed
True
"""

from __future__ import annotations

__version__ = "0.1.0"

# units
from .units import (
    P_STD, ATM,
    SCCM_TO_PA_M3_S, SCCS_TO_PA_M3_S, MBAR_L_S_TO_PA_M3_S, ATM_CC_S_TO_PA_M3_S,
    sccm_to_pa_m3_s, sccs_to_pa_m3_s, mbar_l_s_to_pa_m3_s, atm_cc_s_to_pa_m3_s,
    pa_m3_s_to_sccm, pa_m3_s_to_sccs, pa_m3_s_to_mbar_l_s, pa_m3_s_to_atm_cc_s,
    sccm_to_mbar_l_s, mbar_l_s_to_sccm,
    convert, all_units,
)

# physics
from .physics import (
    conductance_from_leak_rate, leak_rate_from_conductance, time_constant,
    pressure_decay, delta_p_over_window, leak_rate_from_decay,
    leak_rate_exact_window, apparent_dp_from_dt, temperature_corrected_pressure,
    simulate_decay, DecayResult,
)

# temperature
from .temperature import (
    ThermalTransient, compensate, apparent_leak_from_thermal,
)

# instrument
from .instrument import Transducer

# sequence
from .sequence import (
    Phase, SequenceConfig, MeasurementResult, run_sequence,
    settle_error_tradeoff,
)

# decision
from .decision import (
    Verdict, DecisionConfig, Decision, decide, guard_band_from_uncertainty,
)

# flow (stretch)
from .flow import (
    Gas, AIR, HELIUM, mean_free_path, knudsen_number, flow_regime,
    poiseuille_conductance, molecular_conductance,
    viscous_correlation, molecular_correlation, correlate_leak_rate,
)

# gage R&R (stretch)
from .gage_rr import (
    GageRRResult, monte_carlo_leak_measurement, noise_to_leak_rate_sigma,
)

__all__ = [
    "__version__",
    # units
    "P_STD", "ATM",
    "SCCM_TO_PA_M3_S", "SCCS_TO_PA_M3_S", "MBAR_L_S_TO_PA_M3_S", "ATM_CC_S_TO_PA_M3_S",
    "sccm_to_pa_m3_s", "sccs_to_pa_m3_s", "mbar_l_s_to_pa_m3_s", "atm_cc_s_to_pa_m3_s",
    "pa_m3_s_to_sccm", "pa_m3_s_to_sccs", "pa_m3_s_to_mbar_l_s", "pa_m3_s_to_atm_cc_s",
    "sccm_to_mbar_l_s", "mbar_l_s_to_sccm", "convert", "all_units",
    # physics
    "conductance_from_leak_rate", "leak_rate_from_conductance", "time_constant",
    "pressure_decay", "delta_p_over_window", "leak_rate_from_decay",
    "leak_rate_exact_window", "apparent_dp_from_dt", "temperature_corrected_pressure",
    "simulate_decay", "DecayResult",
    # temperature
    "ThermalTransient", "compensate", "apparent_leak_from_thermal",
    # instrument
    "Transducer",
    # sequence
    "Phase", "SequenceConfig", "MeasurementResult", "run_sequence",
    "settle_error_tradeoff",
    # decision
    "Verdict", "DecisionConfig", "Decision", "decide", "guard_band_from_uncertainty",
    # flow
    "Gas", "AIR", "HELIUM", "mean_free_path", "knudsen_number", "flow_regime",
    "poiseuille_conductance", "molecular_conductance",
    "viscous_correlation", "molecular_correlation", "correlate_leak_rate",
    # gage R&R
    "GageRRResult", "monte_carlo_leak_measurement", "noise_to_leak_rate_sigma",
]
