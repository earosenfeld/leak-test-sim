"""Unit-conversion tests: exact constants + round-trips."""

import numpy as np
import pytest

from leak_test_sim import units


def test_exact_constants():
    # 1 sccm = 101325 Pa * 1e-6 m^3 / 60 s
    assert units.SCCM_TO_PA_M3_S == pytest.approx(1.6888e-3, rel=1e-4)
    assert units.SCCM_TO_PA_M3_S == pytest.approx(101325.0 * 1e-6 / 60.0, rel=0, abs=0)
    # 1 mbar*L/s = 100 Pa * 1e-3 m^3 = 0.1 Pa*m^3/s
    assert units.MBAR_L_S_TO_PA_M3_S == pytest.approx(0.1, rel=0, abs=1e-15)
    # 1 scc/s = 60 sccm
    assert units.SCCS_TO_PA_M3_S == pytest.approx(60.0 * units.SCCM_TO_PA_M3_S, rel=1e-12)
    # atm*cc/s == scc/s here (same standard pressure)
    assert units.ATM_CC_S_TO_PA_M3_S == pytest.approx(0.101325, rel=0, abs=1e-9)


def test_sccm_si_roundtrip():
    for q in [0.0, 1.0, 5.0, 123.456, 1e4]:
        si = units.sccm_to_pa_m3_s(q)
        assert units.pa_m3_s_to_sccm(si) == pytest.approx(q, rel=1e-12, abs=1e-15)


def test_mbar_l_s_si_roundtrip():
    for q in [0.0, 0.1, 1.0, 42.0]:
        si = units.mbar_l_s_to_pa_m3_s(q)
        assert units.pa_m3_s_to_mbar_l_s(si) == pytest.approx(q, rel=1e-12, abs=1e-15)


def test_sccs_si_roundtrip():
    for q in [0.0, 1.0, 9.99]:
        si = units.sccs_to_pa_m3_s(q)
        assert units.pa_m3_s_to_sccs(si) == pytest.approx(q, rel=1e-12, abs=1e-15)


def test_known_cross_conversion():
    # 60 sccm = 1 scc/s = 0.101325 Pa*m^3/s
    assert units.sccm_to_pa_m3_s(60.0) == pytest.approx(0.101325, rel=1e-9)
    # 0.1 Pa*m^3/s = 1 mbar*L/s
    assert units.pa_m3_s_to_mbar_l_s(0.1) == pytest.approx(1.0, rel=1e-12)
    # 1 mbar*L/s in sccm: 0.1 / 1.6888e-3
    assert units.mbar_l_s_to_sccm(1.0) == pytest.approx(0.1 / units.SCCM_TO_PA_M3_S, rel=1e-12)


def test_convert_dispatch_roundtrip():
    for src in ["sccm", "mbar_l_s", "scc/s", "pa_m3_s", "atm_cc_s"]:
        for dst in ["sccm", "mbar_l_s", "scc/s", "pa_m3_s", "atm_cc_s"]:
            v = 7.5
            out = units.convert(v, src, dst)
            back = units.convert(out, dst, src)
            assert back == pytest.approx(v, rel=1e-10)


def test_convert_unknown_unit_raises():
    with pytest.raises(ValueError):
        units.convert(1.0, "furlongs", "sccm")
    with pytest.raises(ValueError):
        units.convert(1.0, "sccm", "fortnights")


def test_all_units_consistency():
    q_si = 0.05
    d = units.all_units(q_si)
    assert d["pa_m3_s"] == pytest.approx(q_si)
    assert d["mbar_l_s"] == pytest.approx(0.5)
    assert d["sccm"] == pytest.approx(q_si / units.SCCM_TO_PA_M3_S)


def test_array_input():
    arr = np.array([1.0, 2.0, 3.0])
    out = units.sccm_to_pa_m3_s(arr)
    assert np.allclose(out, arr * units.SCCM_TO_PA_M3_S)
