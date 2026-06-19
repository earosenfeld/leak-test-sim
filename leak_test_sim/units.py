"""Leak-rate unit conversions.

The SI leak rate is a *throughput* (pressure * volume / time):

    [Pa * m^3 / s]

Industry uses several practical units. The conversions below are *exact* by
the definitions that leak-test instruments (CTS, ATEQ, Zaxis) and the vacuum
community use:

    1 sccm   = standard cubic centimetre per minute
             = P_std * V / t  with P_std = 101325 Pa, V = 1e-6 m^3, t = 60 s
             = 101325 * 1e-6 / 60  Pa*m^3/s
             = 1.688750e-3 Pa*m^3/s

    1 scc/s  = standard cubic centimetre per second   (= 60 sccm)
             = 101325 * 1e-6 Pa*m^3/s
             = 0.101325 Pa*m^3/s

    1 mbar*L/s = 100 Pa * 1e-3 m^3 / s
               = 0.1 Pa*m^3/s

The "standard" pressure baked into sccm is 101325 Pa (1 atm). Some vendors use
100 kPa or 14.696 psia bases; we expose ``P_STD`` so the convention is explicit
rather than hidden in a magic number.

All public functions are pure: scalar in -> scalar out (and they also accept
numpy arrays element-wise).
"""

from __future__ import annotations

# --- reference constants (SI) -------------------------------------------------

P_STD: float = 101325.0  # Pa, standard pressure used to define "standard cc"
ATM: float = 101325.0    # Pa, 1 standard atmosphere

# --- exact conversion factors to the SI base (Pa*m^3/s) ----------------------

# 1 sccm  -> Pa*m^3/s
SCCM_TO_PA_M3_S: float = P_STD * 1e-6 / 60.0          # 1.6887500e-3
# 1 scc/s -> Pa*m^3/s  (= 60 sccm)
SCCS_TO_PA_M3_S: float = P_STD * 1e-6                 # 0.101325
# 1 mbar*L/s -> Pa*m^3/s
MBAR_L_S_TO_PA_M3_S: float = 100.0 * 1e-3             # 0.1
# 1 atm*cc/s -> Pa*m^3/s  (another common vendor unit)
ATM_CC_S_TO_PA_M3_S: float = ATM * 1e-6               # 0.101325


# --- to SI base ---------------------------------------------------------------

def sccm_to_pa_m3_s(q_sccm):
    """Convert a leak rate in sccm to Pa*m^3/s."""
    return q_sccm * SCCM_TO_PA_M3_S


def sccs_to_pa_m3_s(q_sccs):
    """Convert a leak rate in standard cc/s to Pa*m^3/s."""
    return q_sccs * SCCS_TO_PA_M3_S


def mbar_l_s_to_pa_m3_s(q_mbar_l_s):
    """Convert a leak rate in mbar*L/s to Pa*m^3/s."""
    return q_mbar_l_s * MBAR_L_S_TO_PA_M3_S


def atm_cc_s_to_pa_m3_s(q_atm_cc_s):
    """Convert a leak rate in atm*cc/s to Pa*m^3/s."""
    return q_atm_cc_s * ATM_CC_S_TO_PA_M3_S


# --- from SI base -------------------------------------------------------------

def pa_m3_s_to_sccm(q_si):
    """Convert a leak rate in Pa*m^3/s to sccm."""
    return q_si / SCCM_TO_PA_M3_S


def pa_m3_s_to_sccs(q_si):
    """Convert a leak rate in Pa*m^3/s to standard cc/s."""
    return q_si / SCCS_TO_PA_M3_S


def pa_m3_s_to_mbar_l_s(q_si):
    """Convert a leak rate in Pa*m^3/s to mbar*L/s."""
    return q_si / MBAR_L_S_TO_PA_M3_S


def pa_m3_s_to_atm_cc_s(q_si):
    """Convert a leak rate in Pa*m^3/s to atm*cc/s."""
    return q_si / ATM_CC_S_TO_PA_M3_S


# --- convenience cross conversions -------------------------------------------

def sccm_to_mbar_l_s(q_sccm):
    return pa_m3_s_to_mbar_l_s(sccm_to_pa_m3_s(q_sccm))


def mbar_l_s_to_sccm(q_mbar_l_s):
    return pa_m3_s_to_sccm(mbar_l_s_to_pa_m3_s(q_mbar_l_s))


# --- generic dispatch ---------------------------------------------------------

_TO_SI = {
    "pa_m3_s": 1.0,
    "pa.m3/s": 1.0,
    "sccm": SCCM_TO_PA_M3_S,
    "sccs": SCCS_TO_PA_M3_S,
    "scc/s": SCCS_TO_PA_M3_S,
    "mbar_l_s": MBAR_L_S_TO_PA_M3_S,
    "mbar.l/s": MBAR_L_S_TO_PA_M3_S,
    "atm_cc_s": ATM_CC_S_TO_PA_M3_S,
}


def convert(value, from_unit: str, to_unit: str):
    """Convert ``value`` between any two supported leak-rate units.

    Supported unit keys (case-insensitive): ``pa_m3_s``, ``sccm``, ``sccs``
    (alias ``scc/s``), ``mbar_l_s``, ``atm_cc_s``.
    """
    f = from_unit.strip().lower()
    t = to_unit.strip().lower()
    if f not in _TO_SI:
        raise ValueError(f"unknown from_unit {from_unit!r}; known: {sorted(_TO_SI)}")
    if t not in _TO_SI:
        raise ValueError(f"unknown to_unit {to_unit!r}; known: {sorted(_TO_SI)}")
    return value * _TO_SI[f] / _TO_SI[t]


def all_units(q_si: float) -> dict[str, float]:
    """Return a leak rate (given in Pa*m^3/s) expressed in every unit.

    Handy for reporting a single measured leak rate in all common units at once.
    """
    return {
        "pa_m3_s": q_si,
        "mbar_l_s": pa_m3_s_to_mbar_l_s(q_si),
        "sccm": pa_m3_s_to_sccm(q_si),
        "sccs": pa_m3_s_to_sccs(q_si),
        "atm_cc_s": pa_m3_s_to_atm_cc_s(q_si),
    }
