# ab-test-integrity-kit

**An A/B test integrity checker whose readout refuses to say "ship" when the experiment is broken: SRM detection for two and multi-arm tests, a Monte Carlo demonstration that peeking triples the false-positive rate (and that an O'Brien-Fleming correction restores it), and CUPED variance reduction — all behind a single gate that returns SHIP, DO_NOT_SHIP, or INCONCLUSIVE.**

Most experiment readouts print an effect and a p-value and trust the reader to ask the awkward questions. The awkward questions are the whole game: was the assignment actually random, did anyone peek, and did a metric that was supposed to stay flat quietly break? This kit asks all three *before* it believes the effect, and when one fails, the readout does not print a warning next to the ship verdict — it **refuses the verdict**. That gate is what makes it a kit rather than a notebook.

> **Synthetic data only.** Every figure is fabricated by a seeded generator for a fictional grocery-subscription store (**Northwind Pantry**). No real company, customer, or revenue data is used anywhere in this project.

Stdlib-only Python — deliberately. An integrity check is an audit artifact: anyone should be able to read every line of the statistics, re-run it offline, and get byte-identical outputs, with no dependency tree between them and the math. Zero network, zero API keys, zero external dependencies. There is deliberately **no power / minimum-detectable-effect planner here** — sizing a test before it runs is a different job, covered by sibling builds; this kit judges a test that already ran.

## Quickstart

```bash
# 1. Generate a synthetic experiment (4,000 users, seed 42).
#    --scenario: clean | srm (planted 70/30 assignment bug)
#                | guardrail (planted checkout-error breach)
python3 cli.py generate --scenario clean --out dataset.jsonl

# 2. SRM check on raw arm counts (any arm count)
python3 cli.py srm --observed 2782,1218 --expected 0.5,0.5

# 3. Peeking Monte Carlo: A/A false positives, raw vs corrected
python3 cli.py peeking

# 4. CUPED: theta + variance reduction on the generated data
python3 cli.py cuped --input dataset.jsonl

# 5. Gated readout of an experiment file (JSON)
python3 cli.py readout --input dataset.jsonl

# 6. Full report: SRM + peeking + CUPED + all scenario readouts
python3 cli.py report

# 7. CI gate: exit 0 = pass, 1 = readout refused (SRM / uncorrected
#    peeking / guardrail breach), 2 = data-quality / IO error
python3 cli.py check --input dataset.jsonl
python3 cli.py check --input dataset.jsonl --looks 5 --uncorrected

# 8. One-command end-to-end demo (incl. localhost serve round-trip)
python3 demo.py

# 9. Tiny localhost API: POST /readout, GET /report, GET /health
python3 cli.py serve --port 8765
```

## The three checks

### 1. SRM — is the split the split you configured?

An **SRM** — sample-ratio mismatch, when the share of users in each arm is far from the configured split — almost never means the randomiser got unlucky. It means something in the pipeline treated the arms differently: a redirect bug, a bot filter, a logging join that drops one variant. Whatever it was, the arms are no longer comparable, so the effect estimate is not evidence.

The test is Pearson's **chi-square** (a sum of squared observed-vs-expected gaps, scaled by the expected counts) goodness-of-fit statistic, with degrees of freedom = arms − 1 and the p-value from the chi-square **survival function** — the probability of a mismatch at least this large under honest assignment. That survival function is a **regularised incomplete gamma function** — a standard special function that turns the chi-square statistic into a tail probability — implemented here in ~30 lines of stdlib (series expansion + Lentz continued fraction, the Numerical Recipes pair). One code path serves every arm count; there is no two-arm shortcut. It is pinned against published critical values for df = 1, 2, 3, 10 (eval E2).

**Threshold: p < 0.001, not 0.05.** The SRM check runs on *every* experiment a team ships. At 0.05, one healthy experiment in twenty would trip the alarm, and an alarm that cries wolf weekly gets muted. At 0.001 the alarm is rare enough to stop the line for — and a genuine assignment bug usually produces a mismatch so large (the planted one below sits at p ≈ 10⁻¹³⁵) that the stricter line loses nothing. Eval E6 pins the discipline: a split with p ≈ 0.0047 — "significant" at 0.05 — does **not** fire.

Seeded results (seed 42):

| Scenario | Counts | chi2 | p-value | Fires? |
|---|---|---|---|---|
| Clean 2-arm | 1,985 / 2,015 | 0.225 | 0.635 | No |
| Planted SRM 2-arm (configured 50/50, delivered ~70/30) | 2,782 / 1,218 | 611.524 | 5.2 × 10⁻¹³⁵ | **Yes** |
| Clean 3-arm | 1,330 / 1,340 / 1,330 | 0.05 | 0.975 | No |
| Planted SRM 3-arm | 1,200 / 1,150 / 1,650 | 113.75 | 2.0 × 10⁻²⁵ | **Yes** |

### 2. Peeking — did repeated looking inflate the false-positive rate?

**Peeking** — re-checking a running experiment's p-value and stopping when it first crosses 0.05 — gives noise five separate chances to look like signal. This kit demonstrates the damage instead of asserting it: a seeded **Monte Carlo** simulation (repeated random simulation of the same experiment) runs 2,000 **A/A tests** — experiments where both arms are identical, so every "significant" result is by definition a false positive — with 5 equally spaced looks:

| Rule | False positives | Rate |
|---|---|---|
| Test every look at p < 0.05, stop at first hit | 278 / 2,000 | **0.139** |
| O'Brien-Fleming corrected looks | 91 / 2,000 | **0.0455** |

The uncorrected rate is ~2.8× the promised 5%. The fix is **alpha spending** — pre-committing how much of the total false-positive budget each look may spend. The **O'Brien-Fleming** shape spends almost nothing early (look 1 requires |z| > 4.56, a nominal p of 5 × 10⁻⁶) and almost everything at the final look (|z| > 2.04, close to the familiar 1.96), so early stopping demands overwhelming evidence while the final read keeps most of its power. The critical values are the standard tabulated boundaries for two-sided alpha = 0.05 (Jennison & Turnbull), each converted to a per-look p-value threshold with `statistics.NormalDist`; look counts beyond 5 use a labelled Lan-DeMets-style approximation. The golden eval states its tolerance honestly: the corrected rate must land within ±0.02 of alpha (observed 0.0455, E9), and both counts are pinned exactly (E8/E9). The simulation draws sequential z-statistics from their exact joint distribution under the null (cumulative standard-normal increments over √k) — the looks are correlated, as real peeks are, rather than five independent coin flips.

### 3. CUPED — can pre-period data make the readout sharper?

**CUPED** — Controlled-experiment Using Pre-Experiment Data — adjusts each user's outcome by a **covariate** (a measurement from *before* the experiment, here pre-period spend, which the treatment cannot have influenced):

```
theta       = Cov(Y, X) / Var(X)
Y_adjusted  = Y − theta · (X − mean(X))
```

The adjustment cannot bias the effect estimate (the covariate has the same distribution in both arms) but removes the outcome variance the covariate explains — a factor of (1 − rho²), where **rho** is the covariate–outcome correlation. The generator plants rho = 0.7, so the expected reduction is rho² = 0.49. On the seeded sample: **theta = 0.70491, observed rho = 0.689, variance falls from 103.18 to 54.18 — a reduction of 0.4749**, equal to the sample rho² exactly (an algebraic identity the evals pin, E10) and within 0.015 of the planted 0.49. At rho = 0 the same code returns theta ≈ 0 and changes nothing (E11) — CUPED that "helps" an uncorrelated covariate would be inventing precision.

## The readout gate

`evaluate_readout` (and `cli.py readout` / `check`) assembles the verdict in a fixed order:

1. **SRM fires → DO_NOT_SHIP.** The arms are not comparable; the effect is not evidence.
2. **Peeked without correction (`looks > 1`, uncorrected) → DO_NOT_SHIP.** The quoted p-value understates the false-positive risk ~3×.
3. **Guardrail breach → DO_NOT_SHIP.** A **guardrail metric** — a metric that must not get worse, here the checkout error rate — may not exceed the control arm's rate by more than the margin (default 0.01). The planted guardrail scenario breaches by **0.0345** (5.26% vs 1.81%) despite a genuine primary lift.
4. Otherwise the primary test (two-sample z on the CUPED-adjusted outcome) decides: positive and significant at alpha = 0.05 → **SHIP**; anything else → **INCONCLUSIVE**.

Seeded verdicts: clean → **SHIP** (CUPED-adjusted difference +3.514, SE 0.226 with CUPED vs 0.315 without); planted SRM, planted guardrail, and clean-but-peeked-uncorrected → **DO_NOT_SHIP**, each with its reasons printed.

## Architecture

| Module | Job |
|---|---|
| `statsmath.py` | Regularised incomplete gamma (series + continued fraction), chi-square survival function, normal CDF/quantiles via `statistics.NormalDist` |
| `srm.py` | Chi-square SRM test for any arm count ≥ 2, threshold p < 0.001 |
| `peeking.py` | O'Brien-Fleming boundaries; seeded A/A Monte Carlo of uncorrected vs corrected peeking |
| `cuped.py` | theta from pre-period covariance, adjusted outcomes, variance-reduction accounting |
| `generator.py` | Deterministic synthetic Northwind Pantry experiments (seed 42, 4,000 users); `clean` / `srm` / `guardrail` planted scenarios; CSV / JSONL / JSON IO |
| `validation.py` | Data-quality gate: required fields, types (incl. the bool-is-an-int trap), duplicate users, ≥ 2 arms — reports issues, never raises |
| `readout.py` | The refusal gate: SRM + peeking + guardrail checks, then the CUPED-adjusted primary z-test → SHIP / DO_NOT_SHIP / INCONCLUSIVE |
| `report.py` | Full report assembly: both SRM arm counts, peeking calibration, CUPED, and all scenario readouts |
| `cli.py` | `generate` / `srm` / `peeking` / `cuped` / `readout` / `report` / `check` (CI exit 0/1/2) / `demo` / `serve` (stdlib `http.server` on 127.0.0.1: `POST /readout`, `GET /report`, `GET /health`) |
| `demo.py` | End-to-end narrative with asserts: determinism, planted SRM fires, peeking inflation + correction, CUPED ≈ rho², gate verdicts, check exit codes, live localhost serve round-trip |
| `fixtures/` | The seeded clean dataset (`sample_experiment.jsonl` + `.csv`) with pinned totals |

## Evals

`python3 evals/run_evals.py` → **16/16 golden evals**, report committed at `evals/eval_report.json` (md5 `655ad45c925404d7992683b2ac96c84d`, byte-identical across 3 verified runs):

- **E1** generator determinism — seed 42 JSONL bytes pinned by md5, counts 1,985 / 2,015
- **E2** chi-square survival function at published critical values (df 1/2/3/10 → 0.05), df=2 closed form exp(−x/2), gamma vs `erfc`
- **E3** SRM clean two-arm pinned (p = 0.635256, no fire; perfect split p = 1)
- **E4** SRM planted two-arm pinned (chi2 = 611.524, p < 10⁻¹⁰⁰, fires)
- **E5** SRM multi-arm — clean chi2 = 0.05 no fire; planted chi2 = 113.75 fires (df = 2, same gamma path)
- **E6** threshold discipline — p ≈ 0.0047 does not fire the 0.001 line; p ≈ 7.7 × 10⁻⁶ does
- **E7** O'Brien-Fleming boundaries pinned (K=5 z values + NormalDist nominal p's)
- **E8** peeking uncorrected false positives pinned (278/2,000 = 0.139)
- **E9** peeking corrected false positives pinned (91/2,000 = 0.0455, within the stated ±0.02 of alpha)
- **E10** CUPED seeded pinned (theta = 0.70491, reduction 0.474941 = sample rho², within 0.03 of planted 0.49)
- **E11** CUPED hand-computed (perfect line → theta 2, reduction 1; uncorrelated → theta 0, reduction 0)
- **E12** gate verdicts (clean SHIP; SRM / guardrail / peeked DO_NOT_SHIP; peeked-corrected SHIP)
- **E13** CLI `check` exit codes 0 / 1 / 1 / 2
- **E14** serve round-trip matches local readout exactly
- **E15** bundled fixture pinned (4,000 rows, total outcome 206,141.52, validates clean, SHIP)
- **E16** full report deterministic, sections pinned

Unit tests: `python3 -m unittest discover -s tests` → **108 hermetic tests**, including a no-network enforcement test (sockets stubbed), hand-computed chi-square/CUPED cases, gamma identities against `math.erfc` and the exponential closed form, boundary monotonicity, gate-refusal combinations, the control-first arm-ordering regression, validation defects, CLI exit codes, JSON/CSV round-trips, and serve round-trips.

## Development loop — what the checks actually caught

1. **The gate compared the effect backwards, and the demo caught it.** The first readout ordered arms by first appearance — and the seeded dataset's first user happens to be in treatment, so "control" silently became arm B and the planted +3.5 effect read out as −3.5 (verdict: INCONCLUSIVE on a textbook SHIP). The demo's stage-5 assert failed before any test did. Fix: an arm named `control` is always the baseline, wherever its first row falls; a regression test now pins both the ordering and the effect sign.
2. **Zero variance read as p = 1, and a unit test caught it.** On a hand-built dataset where the outcome is an exact linear function of the covariate, CUPED drives the within-arm variance to exactly zero; the two-sample z guarded `se == 0` by returning z = 0 — reporting the strongest possible evidence as *no evidence at all*. Fix: zero standard error with a non-zero difference is z = ±∞ (p = 0); zero difference still gives p = 1.
3. **Validated design decision: pooled theta.** CUPED's theta is estimated on the pooled two-arm sample rather than the control arm alone. That is safe here precisely because the covariate is pre-period — treatment assignment cannot change its relationship to the outcome retroactively — and pooling halves theta's own sampling noise. It is stated in `readout.py` so the choice is auditable rather than buried.

## Scope and honest limits

- **The peeking simulation uses the normal approximation.** Sequential z-statistics are drawn from their exact joint null distribution instead of simulating per-user Bernoulli conversions look by look. The correlation structure between looks — the thing that makes peeking dangerous — is exact; per-user discreteness at small samples is not modelled.
- **O'Brien-Fleming values are tabulated for K ≤ 5 looks at alpha = 0.05.** Other alphas scale the tabulated z values (exact at 0.05); K > 5 uses a labelled Lan-DeMets-style approximation, not the recursive numerical integration a production group-sequential library would run.
- **The guardrail rule is a margin on a difference of rates**, not a sequential test: a guardrail that is itself peeked at daily needs its own error budget, which this kit does not give it.
- **SRM says the split is wrong, never why.** Randomiser bugs, redirect asymmetries, and join losses all look identical to a chi-square test; the kit's job is to stop the readout, not to debug the pipeline.
- **CUPED's gain is capped by the covariate you actually have.** A pre-period metric correlated at rho = 0.3 buys a 9% variance reduction, not the 47% planted here; the eval's tidy number is a property of the tidy plant (rho = 0.7).
- **The primary test is a z-test on CUPED-adjusted outcomes** with theta estimated on the same data (standard practice; the small theta-estimation uncertainty is not propagated into the standard error).
- **The API is a demo surface.** It binds 127.0.0.1 only, holds its dataset in memory, and resets on restart.

## Why this project

Experimentation platforms at scale (the teams behind Coinbase's and Anthropic's experimentation and causal-measurement work, and risk teams like Upstart's that live or die by holdout discipline) treat integrity checks as infrastructure, not etiquette: an SRM alarm pages someone, an uncorrected peek invalidates a launch review, a guardrail breach blocks a rollout. This project rebuilds that discipline as a small, inspectable analytics-engineering artifact — the statistics implemented from first principles in the standard library, every claim calibrated by a seeded simulation or pinned against a published value, and a readout that would rather refuse than bless a broken experiment.
