#!/usr/bin/env python3
"""End-to-end pressure-decay leak test: fill -> settle -> test -> exhaust.

Runs a realistic cycle on a part with a known leak plus a fill temperature
transient and instrument noise, then prints the pass/fail verdict with the leak
rate in every unit -- both uncompensated and temperature-compensated -- the way
a CTS/ATEQ/Zaxis station would report it. Finally it demonstrates the
settle-time vs error tradeoff, a Gage R&R study, and air<->helium correlation.

Run:  python examples/run_leak_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly from a fresh clone (no install) as `python examples/...`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from leak_test_sim import (
    SequenceConfig, run_sequence, Transducer, ThermalTransient,
    DecisionConfig, decide, guard_band_from_uncertainty,
    conductance_from_leak_rate, sccm_to_pa_m3_s, all_units,
    settle_error_tradeoff, monte_carlo_leak_measurement,
    AIR, HELIUM, correlate_leak_rate, knudsen_number, flow_regime,
)


def fmt_units(q_si: float) -> str:
    u = all_units(q_si)
    return (f"{u['sccm']:.4f} sccm | {u['mbar_l_s']:.3e} mbar.L/s | "
            f"{u['pa_m3_s']:.3e} Pa.m3/s | {u['sccs']:.3e} scc/s")


def hr(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    # ---- part / test definition ---------------------------------------------
    V = 1.0e-4                    # 100 cc sealed test volume
    P_test = 300000.0            # 3 bar absolute (~2 barg) test pressure
    Q_true = sccm_to_pa_m3_s(4.0)  # the part has a real 4 sccm leak
    C = conductance_from_leak_rate(Q_true, P_test)

    reject_sccm = 5.0            # parts leaking >= 5 sccm are rejects
    reject_limit = sccm_to_pa_m3_s(reject_sccm)

    cfg = SequenceConfig(
        test_pressure=P_test, volume=V,
        fill_time=3.0, settle_time=8.0, test_time=5.0, exhaust_time=2.0,
        dt=0.005,
    )

    # fill heating: gas starts 4 K hot and relaxes (tau=8 s)
    thermal = ThermalTransient(T_ambient=293.15, dT0=4.0, tau_thermal=8.0)
    # transducer: 1 Pa resolution, 5 Pa RMS noise, slight baseline drift
    tx = Transducer(resolution=1.0, noise_std=5.0, drift_rate=0.3, seed=7)

    hr("PART & TEST SETUP")
    print(f"  Test volume V      : {V*1e6:.1f} cc ({V:.2e} m^3)")
    print(f"  Test pressure      : {P_test/1000:.1f} kPa abs "
          f"({(P_test-101325)/1000:.1f} kPa gauge)")
    print(f"  True leak (planted): {fmt_units(Q_true)}")
    print(f"  Reject limit       : {reject_sccm:.1f} sccm "
          f"({reject_limit:.3e} Pa.m3/s)")
    print(f"  Decay time const   : tau = V/C = {V/C:.1f} s")
    print(f"  Sequence           : fill {cfg.fill_time}s -> settle "
          f"{cfg.settle_time}s -> test {cfg.test_time}s -> exhaust "
          f"{cfg.exhaust_time}s")

    # ---- run the cycle (with thermal + noise) -------------------------------
    res = run_sequence(cfg, C=C, temp_profile=thermal, T_ref=293.15,
                       transducer=tx)

    hr("MEASUREMENT (fill -> settle -> test -> exhaust)")
    print(f"  TEST window: P_start={res.P_start:.1f} Pa  P_end={res.P_end:.1f} Pa")
    print(f"  dP over {res.dt_test:.0f}s test window: {res.dP:.2f} Pa")
    print()
    print("  UNCOMPENSATED leak rate:")
    print(f"    {fmt_units(abs(res.leak_rate_si))}")
    print("  TEMPERATURE-COMPENSATED leak rate  (P_corr = P*(T_ref/T)):")
    print(f"    {fmt_units(abs(res.leak_rate_si_compensated))}")

    # ---- pass/fail with guard band ------------------------------------------
    # size the guard band from a quick gauge-uncertainty estimate
    gauge_sigma = monte_carlo_leak_measurement(
        cfg, tx, C=C, tolerance=reject_limit, n=200, temp_profile=thermal,
        seed=1).sigma
    guard = guard_band_from_uncertainty(gauge_sigma, k=2.0)
    dcfg = DecisionConfig(reject_limit=reject_limit, guard_band=guard)

    d_uncomp = decide(abs(res.leak_rate_si), dcfg)
    d_comp = decide(abs(res.leak_rate_si_compensated), dcfg)

    hr("PASS / FAIL DECISION  (guard band = 2 sigma)")
    print(f"  Guard band          : {guard:.3e} Pa.m3/s "
          f"({guard/sccm_to_pa_m3_s(1.0):.3f} sccm)")
    print(f"  Accept threshold    : {d_comp.accept_threshold:.3e} Pa.m3/s")
    print(f"  UNCOMPENSATED verdict: {d_uncomp.verdict.value.upper()}  "
          f"(thermal transient biases this -- likely false reject)")
    print(f"  COMPENSATED   verdict: {d_comp.verdict.value.upper()}  "
          f"(true ~4 sccm leak, below 5 sccm reject -> good part)")

    # ---- settle-time vs error tradeoff --------------------------------------
    hr("SETTLE-TIME vs MEASUREMENT ERROR  (pure thermal, zero real leak)")
    rows = settle_error_tradeoff(
        SequenceConfig(test_pressure=P_test, volume=V, test_time=5.0, dt=0.01),
        thermal, C=0.0, settle_times=[1.0, 2.0, 5.0, 10.0, 20.0, 40.0])
    print(f"  {'settle(s)':>10} {'false leak (sccm)':>20} {'dT in window (K)':>18}")
    for r in rows:
        false_sccm = abs(r["leak_rate_si"]) / sccm_to_pa_m3_s(1.0)
        print(f"  {r['settle_time']:>10.0f} {false_sccm:>20.4f} "
              f"{r['dT_over_window']:>18.4f}")
    print("  -> too-short settle leaves a cooling transient that fakes a leak.")

    # ---- Gage R&R -----------------------------------------------------------
    hr("GAGE R&R / MEASUREMENT CAPABILITY  (Monte-Carlo, n=300)")
    tol_band = sccm_to_pa_m3_s(reject_sccm)   # tolerance = reject limit
    grr = monte_carlo_leak_measurement(
        cfg, Transducer(resolution=1.0, noise_std=5.0, seed=0),
        C=C, tolerance=tol_band, n=300, seed=5)
    print(f"  mean measured leak : {grr.mean/sccm_to_pa_m3_s(1.0):.4f} sccm")
    print(f"  repeatability sigma: {grr.sigma:.3e} Pa.m3/s "
          f"({grr.sigma/sccm_to_pa_m3_s(1.0):.4f} sccm)")
    print(f"  %GRR vs tolerance  : {grr.pct_grr:.2f}%   "
          f"({'CAPABLE' if grr.capable else 'review'} -- AIAG <10% good)")
    print(f"  distinct categories: {grr.ndc:.1f}")

    # ---- gas correlation ----------------------------------------------------
    hr("GAS CORRELATION  (air on the line vs helium spec)")
    Kn_small = knudsen_number(50e-9, P_test, gas=AIR)   # 50 nm channel
    print(f"  50 nm channel @ test P: Kn={Kn_small:.2f} -> {flow_regime(Kn_small)} flow")
    q_air = Q_true
    q_he_mol = correlate_leak_rate(q_air, AIR, HELIUM, "molecular")
    q_he_visc = correlate_leak_rate(q_air, AIR, HELIUM, "viscous")
    print(f"  air leak {q_air/sccm_to_pa_m3_s(1.0):.2f} sccm equates to helium:")
    print(f"    molecular regime : {q_he_mol/sccm_to_pa_m3_s(1.0):.2f} sccm "
          f"(He lighter -> leaks faster, x{q_he_mol/q_air:.2f})")
    print(f"    viscous regime   : {q_he_visc/sccm_to_pa_m3_s(1.0):.2f} sccm "
          f"(He more viscous -> leaks slower, x{q_he_visc/q_air:.2f})")

    print()


if __name__ == "__main__":
    main()
