#!/usr/bin/env python3
"""Generate the leak-test-sim visualization figures into ``assets/``.

Every figure is produced from the *real* package API (no fabricated data):
pressure-decay sequence, temperature compensation, Gage R&R, and the
guard-banded accept/reject decision. Run headless (Agg):

    .venv/bin/python scripts/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# --- house style (verbatim) ---------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#334155", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#e2e8f0", "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "legend.frameon": False, "lines.linewidth": 2.0,
})
PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]

# --- real API -----------------------------------------------------------------
from leak_test_sim import (
    SequenceConfig, run_sequence, Transducer, ThermalTransient,
    DecisionConfig, decide, Verdict, guard_band_from_uncertainty,
    conductance_from_leak_rate, sccm_to_pa_m3_s, pa_m3_s_to_sccm,
    monte_carlo_leak_measurement,
)

ASSETS = Path(__file__).resolve().parent.parent / "assets"
ASSETS.mkdir(exist_ok=True)

# phase-band colors (settle/test highlighted; fill/exhaust muted)
BAND = {
    "fill":    "#f1f5f9",
    "settle":  "#fef3c7",
    "test":    "#dbeafe",
    "exhaust": "#f1f5f9",
}


def _sccm(q_si: float) -> float:
    return pa_m3_s_to_sccm(q_si)


def _phase_band(ax, t0, t1, key, label, y_label):
    """Shade [t0, t1] as a labelled background phase band."""
    ax.axvspan(t0, t1, color=BAND[key], zorder=0)
    ax.text((t0 + t1) / 2.0, y_label, label, ha="center", va="top",
            fontsize=9.5, fontweight="bold", color="#475569", zorder=5)


# =============================================================================
# Figure 1 — full fill -> settle -> test -> exhaust pressure sequence
# =============================================================================
def fig_pressure_decay_sequence():
    V = 1.0e-4                       # 100 cc
    P_test = 300_000.0               # 3 bar abs
    P_atm = 101_325.0
    Q_true = sccm_to_pa_m3_s(8.0)    # clearly-leaking part so dP is visible
    C = conductance_from_leak_rate(Q_true, P_test)

    cfg = SequenceConfig(
        test_pressure=P_test, volume=V,
        fill_time=3.0, settle_time=8.0, test_time=6.0, exhaust_time=3.0,
        P_atm=P_atm, dt=0.005,
    )
    res = run_sequence(cfg, C=C)     # decay trace spans SETTLE+TEST (t=0 at settle start)

    # ---- build a full timeline: fill ramp + (settle+test core) + exhaust vent
    t_core = res.decay.t                       # 0 .. settle+test
    P_core = res.decay.P
    t_settle0 = cfg.fill_time
    t_test0 = t_settle0 + cfg.settle_time
    t_test1 = t_test0 + cfg.test_time
    t_exh0 = t_test1
    t_exh1 = t_exh0 + cfg.exhaust_time

    # fill: smooth rise from atmosphere to P_test (cosine ease, illustrative)
    t_fill = np.linspace(0.0, cfg.fill_time, 60)
    ease = 0.5 - 0.5 * np.cos(np.pi * t_fill / cfg.fill_time)
    P_fill = P_atm + (P_test - P_atm) * ease

    # exhaust: fast decay back to atmosphere from the last core pressure
    t_exh = np.linspace(0.0, cfg.exhaust_time, 60)
    P_last = P_core[-1]
    P_exh = P_atm + (P_last - P_atm) * np.exp(-t_exh / (cfg.exhaust_time / 4.0))

    fig, ax = plt.subplots(figsize=(9.6, 5.2))

    # phase bands (label near top of axes in data coords later)
    y_top = (P_test + 6_000) / 1000.0
    _phase_band(ax, 0.0, t_settle0, "fill", "FILL", y_top)
    _phase_band(ax, t_settle0, t_test0, "settle", "SETTLE", y_top)
    _phase_band(ax, t_test0, t_test1, "test", "TEST", y_top)
    _phase_band(ax, t_exh0, t_exh1, "exhaust", "EXHAUST", y_top)

    ax.plot(t_fill, P_fill / 1000.0, color="#94a3b8", lw=2.0)
    ax.plot(t_settle0 + t_core, P_core / 1000.0, color=PALETTE[0], lw=2.4,
            label="isolated decay (measured)")
    ax.plot(t_exh0 + t_exh, P_exh / 1000.0, color="#94a3b8", lw=2.0)

    # mark TEST-window start/end pressures and annotate dP
    Ps, Pe = res.P_start / 1000.0, res.P_end / 1000.0
    ax.plot([t_test0, t_test1], [Ps, Pe], "o", color=PALETTE[1],
            ms=7, zorder=6)
    ax.hlines(Ps, t_test0, t_test1, color=PALETTE[1], lw=1.0, ls="--", zorder=4)

    # dP annotation: leader from the test window out to a white box in the
    # right-hand margin (kept clear of the EXHAUST band / phase labels).
    x_box = t_exh1 + 0.6
    y_box = (P_test / 1000.0) - 35
    ax.annotate(
        f"$\\Delta P$ = {res.dP:.0f} Pa over {cfg.test_time:.0f} s\n"
        f"→ {_sccm(res.leak_rate_si):.1f} sccm measured",
        xy=(t_test1, Pe), xytext=(x_box, y_box),
        va="center", ha="left", fontsize=9.0, color=PALETTE[1],
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=PALETTE[1], lw=1.0),
        arrowprops=dict(arrowstyle="-", color=PALETTE[1], lw=1.2,
                        connectionstyle="arc3,rad=-0.2"))

    ax.axhline(P_atm / 1000.0, color="#cbd5e1", lw=1.0, ls=":")
    ax.text(t_exh1, P_atm / 1000.0 + 1.5, "atmosphere", ha="right",
            va="bottom", fontsize=8.5, color="#94a3b8")

    ax.set_xlim(0, t_exh1 + 3.2)
    ax.set_ylim(P_atm / 1000.0 - 6, P_test / 1000.0 + 12)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("absolute pressure (kPa)")
    ax.set_title("Pressure-decay leak test: fill → settle → test → exhaust")
    ax.legend(loc="lower left")
    fig.text(0.012, 0.01,
             f"V = {V*1e6:.0f} cc   ·   test P = {P_test/1000:.0f} kPa abs"
             f"   ·   ΔP measured only over the isolated TEST window",
             fontsize=8.5, color="#64748b")

    out = ASSETS / "pressure_decay_sequence.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# =============================================================================
# Figure 2 — temperature compensation: zero real leak, fill transient
# =============================================================================
def fig_temperature_compensation():
    V = 1.0e-4
    P_test = 300_000.0
    T_ref = 293.15
    # ZERO real leak; only a fill-heating transient that cools through the window
    thermal = ThermalTransient(T_ambient=T_ref, dT0=4.0, tau_thermal=8.0)
    reject_sccm = 5.0
    reject_limit = sccm_to_pa_m3_s(reject_sccm)

    cfg = SequenceConfig(
        test_pressure=P_test, volume=V,
        fill_time=3.0, settle_time=6.0, test_time=6.0, exhaust_time=2.0,
        dt=0.005,
    )
    # C = 0 => perfectly sealed; any apparent dP is purely thermal
    res = run_sequence(cfg, C=0.0, temp_profile=thermal, T_ref=T_ref)

    t = res.decay.t
    P_unc = res.decay.P
    # compensated trace (same correction the engine uses internally)
    from leak_test_sim import compensate
    P_cmp = compensate(P_unc, res.decay.T, T_ref)

    t_test0 = cfg.settle_time
    t_test1 = cfg.settle_time + cfg.test_time

    fig, (ax, axT) = plt.subplots(
        2, 1, figsize=(9.6, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12})

    # phase bands on the pressure axis. SETTLE label sits low (clear of the
    # legend, top-left); TEST label sits high (clear of the annotation box,
    # bottom-right) -- so the two never collide with either box.
    p_lo = min(P_unc.min(), P_cmp.min()) / 1000.0
    p_hi = max(P_unc.max(), P_cmp.max()) / 1000.0
    ax.axvspan(0.0, t_test0, color=BAND["settle"], zorder=0)
    ax.axvspan(t_test0, t_test1, color=BAND["test"], zorder=0)
    ax.text(t_test0 / 2.0, p_lo + 0.12, "SETTLE", ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", color="#475569")
    ax.text((t_test0 + t_test1) / 2.0, p_hi, "TEST (ΔP window)",
            ha="center", va="top", fontsize=9.5, fontweight="bold",
            color="#475569")

    ax.plot(t, P_unc / 1000.0, color=PALETTE[1], lw=2.4,
            label="uncompensated  (cooling gas → looks like a leak)")
    ax.plot(t, P_cmp / 1000.0, color=PALETTE[0], lw=2.4,
            label="temperature-compensated  $P\\cdot(T_{ref}/T)$  (flat → true)")

    # apparent vs true leak-rate annotation
    q_unc = abs(res.leak_rate_si)
    q_cmp = abs(res.leak_rate_si_compensated)
    d_unc = decide(q_unc, DecisionConfig(reject_limit=reject_limit))
    d_cmp = decide(q_cmp, DecisionConfig(reject_limit=reject_limit))
    txt = (
        f"true leak = 0 (sealed part)\n"
        f"apparent (uncomp.):  {_sccm(q_unc):.2f} sccm  →  "
        f"{d_unc.verdict.value.upper()}\n"
        f"compensated:         {_sccm(q_cmp):.2f} sccm  →  "
        f"{d_cmp.verdict.value.upper()}\n"
        f"(reject limit {reject_sccm:.0f} sccm)")
    ax.text(0.985, 0.04, txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9.0,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cbd5e1"))

    ax.set_ylabel("absolute pressure (kPa)")
    ax.set_title("Temperature compensation rescues a good part from a false reject")
    ax.legend(loc="upper left", fontsize=9.0)

    # temperature subplot
    axT.plot(t, res.decay.T, color=PALETTE[3], lw=2.0)
    axT.axvspan(t_test0, t_test1, color=BAND["test"], zorder=0)
    axT.axhline(T_ref, color="#cbd5e1", ls=":", lw=1.0)
    axT.set_ylabel("gas T (K)")
    axT.set_xlabel("time since settle start (s)")
    axT.text(t_test1, T_ref + 0.05, "$T_{ref}$", ha="left", va="bottom",
             fontsize=8.5, color="#94a3b8")

    out = ASSETS / "temperature_compensation.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# =============================================================================
# Figure 3 — Gage R&R: repeated Monte-Carlo measurements vs reject limit
# =============================================================================
def fig_gage_rr():
    V = 1.0e-4
    P_test = 300_000.0
    Q_true = sccm_to_pa_m3_s(4.0)    # same good part, leaks 4 sccm
    C = conductance_from_leak_rate(Q_true, P_test)
    reject_sccm = 5.0
    reject_limit = sccm_to_pa_m3_s(reject_sccm)

    cfg = SequenceConfig(
        test_pressure=P_test, volume=V,
        fill_time=3.0, settle_time=8.0, test_time=5.0, exhaust_time=2.0,
        dt=0.01,
    )
    tx = Transducer(resolution=1.0, noise_std=5.0, seed=0)
    grr = monte_carlo_leak_measurement(
        cfg, tx, C=C, tolerance=reject_limit, n=400, seed=5)

    samples_sccm = _sccm(grr.samples)
    mean_sccm = _sccm(grr.mean)
    sigma_sccm = _sccm(grr.sigma)

    fig, ax = plt.subplots(figsize=(9.6, 5.2))

    ax.hist(samples_sccm, bins=34, color=PALETTE[0], alpha=0.78,
            edgecolor="white", linewidth=0.6, label="repeated measurements")

    # mean and +/-3 sigma spread
    ax.axvline(mean_sccm, color="#1e3a8a", lw=2.0,
               label=f"mean = {mean_sccm:.2f} sccm")
    for s in (-3, 3):
        ax.axvline(mean_sccm + s * sigma_sccm, color="#1e3a8a", lw=1.0, ls="--")
    ax.axvspan(mean_sccm - 3 * sigma_sccm, mean_sccm + 3 * sigma_sccm,
               color=PALETTE[0], alpha=0.08, zorder=0)

    # reject limit
    ax.axvline(reject_sccm, color=PALETTE[1], lw=2.2,
               label=f"reject limit = {reject_sccm:.0f} sccm")

    cap = "CAPABLE" if grr.capable else "review"
    txt = (
        f"%GRR vs tolerance = {grr.pct_grr:.1f}%   ({cap}, AIAG <10%)\n"
        f"repeatability σ = {sigma_sccm:.3f} sccm\n"
        f"distinct categories (ndc) = {grr.ndc:.0f}\n"
        f"n = {grr.n} Monte-Carlo measurements of one part")
    ax.text(0.985, 0.96, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=9.0,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cbd5e1"))

    ax.set_xlabel("measured leak rate (sccm)")
    ax.set_ylabel("count")
    ax.set_title("Gage R&R: measurement spread of one part vs the reject limit")
    ax.legend(loc="upper left", fontsize=9.0)

    out = ASSETS / "gage_rr.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# =============================================================================
# Figure 4 — guard-banded ACCEPT / RETEST / REJECT decision zones
# =============================================================================
def fig_decision_guardband():
    reject_sccm = 5.0
    reject_limit = sccm_to_pa_m3_s(reject_sccm)
    # size the guard band from a realistic gauge sigma
    sigma_sccm = 0.35
    guard = guard_band_from_uncertainty(sccm_to_pa_m3_s(sigma_sccm), k=2.0)
    guard_sccm = _sccm(guard)
    accept_sccm = reject_sccm - guard_sccm

    cfg = DecisionConfig(reject_limit=reject_limit, guard_band=guard)

    # one example point per zone (classified by the REAL decide())
    examples_sccm = [3.2, accept_sccm + guard_sccm * 0.45, 6.1]
    labels = ["good part", "near limit", "leaker"]

    x_max = 7.5
    fig, ax = plt.subplots(figsize=(9.6, 4.6))

    # zone shading
    ax.axvspan(0, accept_sccm, color=PALETTE[2], alpha=0.16, zorder=0)
    ax.axvspan(accept_sccm, reject_sccm, color=PALETTE[3], alpha=0.18, zorder=0)
    ax.axvspan(reject_sccm, x_max, color=PALETTE[1], alpha=0.16, zorder=0)

    # threshold lines
    ax.axvline(accept_sccm, color=PALETTE[3], lw=1.6, ls="--")
    ax.axvline(reject_sccm, color=PALETTE[1], lw=2.0)

    # zone labels
    ax.text(accept_sccm / 2.0, 0.86, "ACCEPT", ha="center", color="#047857",
            fontweight="bold", fontsize=12)
    ax.text((accept_sccm + reject_sccm) / 2.0, 0.86, "RETEST", ha="center",
            color="#b45309", fontweight="bold", fontsize=12)
    ax.text((reject_sccm + x_max) / 2.0, 0.86, "REJECT", ha="center",
            color="#b91c1c", fontweight="bold", fontsize=12)

    # guard-band bracket between accept threshold and reject limit
    ax.annotate("", xy=(reject_sccm, 0.32), xytext=(accept_sccm, 0.32),
                arrowprops=dict(arrowstyle="<->", color="#b45309", lw=1.6))
    ax.text((accept_sccm + reject_sccm) / 2.0, 0.24,
            f"guard band\n2σ = {guard_sccm:.2f} sccm", ha="center",
            va="top", fontsize=8.5, color="#b45309")

    # example measured points, each placed + classified by real decide()
    zone_color = {Verdict.ACCEPT: PALETTE[2], Verdict.INDETERMINATE: PALETTE[3],
                  Verdict.REJECT: PALETTE[1]}
    for q_sccm, lab in zip(examples_sccm, labels):
        d = decide(sccm_to_pa_m3_s(q_sccm), cfg)
        ax.plot(q_sccm, 0.58, "o", ms=12, color=zone_color[d.verdict],
                markeredgecolor="white", markeredgewidth=1.2, zorder=6)
        ax.annotate(
            f"{lab}\n{q_sccm:.2f} sccm\n→ {d.verdict.value.upper()}",
            xy=(q_sccm, 0.58), xytext=(q_sccm, 0.66), ha="center", va="bottom",
            fontsize=8.5, color="#334155")

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("measured leak rate (sccm)")
    ax.set_title("Guard-banded pass/fail: ACCEPT · RETEST · REJECT zones")
    fig.text(0.012, 0.01,
             f"accept threshold = reject limit − guard band = "
             f"{accept_sccm:.2f} sccm   ·   reject limit = {reject_sccm:.0f} sccm",
             fontsize=8.5, color="#64748b")

    out = ASSETS / "decision_guardband.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    figs = [
        fig_pressure_decay_sequence(),
        fig_temperature_compensation(),
        fig_gage_rr(),
        fig_decision_guardband(),
    ]
    print("Wrote figures:")
    for f in figs:
        kb = f.stat().st_size / 1024.0
        print(f"  {f.relative_to(ASSETS.parent)}  ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
