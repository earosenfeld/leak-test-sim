# leak-test-sim

A physically-correct **pressure-decay & flow leak-test simulator** for manufacturing
quality engineering. It models the full test the way production instruments
(Cincinnati Test Systems, ATEQ, Zaxis) run it — **fill → settle → test → exhaust** —
with ideal-gas pressure decay, the dominant **temperature-transient** false-reject
mechanism and its compensation, instrument noise / resolution / drift, guard-banded
pass/fail, and stretch physics for flow regimes, gas correlation, and Gage R&R.

Everything is validated against **closed-form** leak-test relationships in the test
suite (59 tests). No hand-waving: given a known volume and conductance, the simulator
reproduces the analytic `ΔP`, `τ`, and leak rate to tolerance.

```
fill ──▶ settle ──▶ test ──▶ exhaust
 ▲          ▲         ▲          ▲
 charge   let fill   isolated   vent
 to P     transients ΔP over    to atm
          decay      window
```

---

## The physics

### 1. Pressure decay → leak rate (the core)

A sealed test volume `V` (m³) at absolute pressure `P` (Pa) leaks to atmosphere
`P_atm` through a leak path of **conductance** `C` (m³/s). The molar balance from the
ideal gas law `PV = nRT` (isothermal, fixed `V`) gives a first-order ODE in the
*gauge* pressure:

```
dP/dt = −(C / V) · (P − P_atm)
```

which integrates to an **exponential decay** with time constant `τ = V / C`:

```
P(t) − P_atm = (P0 − P_atm) · exp(−t / τ)
```

The **instantaneous leak rate** (throughput, in Pa·m³/s) is:

```
Q(t) = C · (P(t) − P_atm)        # gas leaving the volume per unit time
```

A pressure-decay instrument doesn't measure `Q` directly. It fills, settles, then
watches the pressure fall by `ΔP` over a fixed **test window** `Δt` and infers:

```
Q_measured ≈ |ΔP| · V / Δt
```

This is **exact in the limit `Δt → 0`** and a small under-estimate for finite `Δt`
(the instrument sees the *average* slope, slightly below the initial `Q0` because the
decay is exponential). `leak_rate_from_decay()` returns the measured value;
`leak_rate_exact_window()` returns the exponential-corrected value so you can compare.

> **Validation** (`tests/test_physics.py`): given `V` and `C`, the simulated `ΔP`
> over `Δt` recovers `Q = |ΔP|·V/Δt` to <1% for a short window, and the ODE
> integrator matches the closed-form `P(t)` to <1 Pa over the whole trace; `τ = V/C`.

### 2. Temperature compensation (the differentiator)

The **dominant false-reject source** in pressure decay is not real leaks — it's
temperature. Filling a part compresses and warms the gas; the walls then pull that
heat away and the gas cools during settle/test. At fixed `V` and `n` the ideal gas
law gives:

```
ΔP / P = ΔT / T          ⇒        ΔP = P · ΔT / T
```

So a cooling gas (`ΔT < 0`) drops pressure **with no real leak** — indistinguishable
from a leak to an uncompensated instrument. A 1 K cool-down on a part at 300 kPa,
293 K fakes `ΔP = 300000 · (−1)/293 ≈ −1024 Pa`, which on a small volume dwarfs the
reject leak rate.

Active **temperature compensation** corrects each pressure reading to a reference
temperature:

```
P_corr(t) = P(t) · (T_ref / T(t))
```

Genuine mass loss survives this correction; pure thermal drift cancels.

> **Validation** (`tests/test_temperature.py`): a pure thermal transient with **zero
> real leak** makes the uncompensated reading exceed the reject limit (false reject),
> while the compensated reading collapses to <1% of it (accept). A second test
> confirms compensation **preserves a genuine leak** rather than erasing it.

### 3. Leak-rate units

The SI leak rate is a throughput, `Pa·m³/s`. Industry uses several units, defined
*exactly* by:

| Unit | Definition | = Pa·m³/s |
|---|---|---|
| `1 sccm` | `101325 Pa × 1e-6 m³ / 60 s` | `1.68875e-3` |
| `1 scc/s` | `101325 Pa × 1e-6 m³` (= 60 sccm) | `0.101325` |
| `1 mbar·L/s` | `100 Pa × 1e-3 m³` | `0.1` |
| `1 atm·cc/s` | `101325 Pa × 1e-6 m³` | `0.101325` |

The "standard" pressure baked into sccm is `P_STD = 101325 Pa` (exposed as a constant,
not a magic number). All conversions round-trip exactly (`tests/test_units.py`).

### 4. Pass/fail with a guard band

Compare the measured leak rate against a **reject limit**. Because the measurement has
uncertainty, a **guard band** pulls the accept threshold in by the gauge uncertainty:

```
measured ≥ reject_limit                 → REJECT          (clearly too leaky)
measured < reject_limit − guard_band    → ACCEPT          (clearly good)
otherwise                               → INDETERMINATE   (retest)
```

Guarding only the accept side is the conservative, ship-no-bad-parts convention used
on production testers. The guard band is typically `k·σ` of the gauge repeatability.

### 5. Instrument realism, flow regimes, Gage R&R (stretch)

- **Transducer**: finite resolution (ADC quantisation), Gaussian noise, baseline
  drift — the real contributors to measurement repeatability.
- **Flow regimes**: classify a leak channel by **Knudsen number** `Kn = λ/d`
  (continuum / transitional / molecular); Poiseuille vs molecular tube conductances.
- **Gas correlation**: air-on-the-line vs helium-spec. Viscous flow `∝ 1/η`
  (air leaks faster, `η_air=1.81e-5`, `η_He=1.96e-5`); molecular flow `∝ 1/√MW`
  (helium leaks ~2.69× faster, `MW_air=28.97`, `MW_He=4.0`).
- **Gage R&R**: Monte-Carlo repeated measurements → repeatability σ, %GRR vs
  tolerance, number of distinct categories (AIAG MSA). A closed-form
  `σ_leak = √2 · σ_noise · V / Δt` cross-checks the Monte-Carlo result.

---

## Worked example

A 100 cc part at 300 kPa absolute with a planted **4 sccm** leak, a 4 K fill-heating
transient (`τ_thermal = 8 s`), and a noisy transducer, tested against a **5 sccm**
reject limit:

```
TEST window: dP over 5 s = −1007 Pa
  UNCOMPENSATED leak rate :  11.93 sccm   →  REJECT   (false reject from cooling)
  COMPENSATED   leak rate :   3.75 sccm   →  ACCEPT   (recovers the true ~4 sccm)

SETTLE-TIME vs ERROR (zero real leak, pure thermal):
   settle(s)   false leak (sccm)
        1            19.6
        5            11.9
       20             1.8
       40             0.15        ← longer settle drives the false reading to zero

GAGE R&R:  σ = 0.082 sccm,  %GRR = 9.87%  (capable),  ndc = 14
GAS CORRELATION (4 sccm air):  molecular → 10.76 sccm He,  viscous → 3.69 sccm He
```

The uncompensated channel **false-rejects a good part** purely from the fill
transient; temperature compensation recovers the true leak and the part passes. This
is the headline value proposition of a real leak-test station, reproduced from first
principles.

Run it:

```bash
python examples/run_leak_test.py
```

---

## Install & run

```bash
cd leak-test-sim
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python numpy scipy pytest matplotlib
uv pip install --python .venv/bin/python -e .

.venv/bin/python -m pytest -q          # 59 tests, all green
.venv/bin/python examples/run_leak_test.py
```

Python ≥ 3.10. Runtime deps: `numpy`, `scipy`. (`matplotlib` only for optional plots.)

## Package layout

```
leak_test_sim/
├── units.py         # exact leak-rate unit conversions (sccm ↔ Pa·m³/s ↔ mbar·L/s ↔ scc/s)
├── physics.py       # pressure decay ODE + closed form, Q↔C, ideal-gas thermal term
├── temperature.py   # fill-heat/cool transient + P_corr = P·(T_ref/T) compensation
├── instrument.py    # transducer resolution / Gaussian noise / baseline drift
├── sequence.py      # fill→settle→test→exhaust state machine + settle/error tradeoff
├── decision.py      # guard-banded accept / reject / indeterminate
├── flow.py          # Knudsen number, Poiseuille vs molecular, air↔He correlation
└── gage_rr.py       # Monte-Carlo measurement capability (%GRR, ndc)
tests/               # 59 closed-form validation tests
examples/run_leak_test.py
```

## Quick API

```python
from leak_test_sim import (
    SequenceConfig, run_sequence, ThermalTransient, Transducer,
    DecisionConfig, decide, conductance_from_leak_rate, sccm_to_pa_m3_s, all_units,
)

cfg = SequenceConfig(test_pressure=300_000.0, volume=1e-4,
                     settle_time=8.0, test_time=5.0)
C = conductance_from_leak_rate(sccm_to_pa_m3_s(4.0), cfg.test_pressure)
thermal = ThermalTransient(dT0=4.0, tau_thermal=8.0)

res = run_sequence(cfg, C=C, temp_profile=thermal, T_ref=293.15,
                   transducer=Transducer(noise_std=5.0, seed=7))

print(all_units(abs(res.leak_rate_si_compensated)))      # leak rate in every unit
print(decide(abs(res.leak_rate_si_compensated),
             DecisionConfig(reject_limit=sccm_to_pa_m3_s(5.0))).verdict)
```

## License

MIT
