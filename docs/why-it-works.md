# Why this package reduces false wins and missed wins

**Audience:** data scientists and ML engineers deciding whether Model A should replace Model B based on eval results. No code reading required.

## 1. Model selection is statistical inference, whether you treat it that way or not

Every "A beats B on the benchmark" decision is a noisy comparison: the benchmark is a finite sample of items, the model is stochastic at temperature > 0, and the observed score gap is an *estimate* of the true gap. Two errors are possible:

- **False win:** you ship A because it scored higher, but the true gap is zero or negative. Cost: a regression in production, plus every downstream decision anchored on a wrong conclusion.
- **Missed win:** A is genuinely better, but the comparison was too noisy to show it, so you discard a real improvement. Cost: the improvement, plus the R&D that produced it.

Standard eval tooling reports a score, and sometimes a standard error on that single score under an independence assumption. Neither error above is about a single score — both are about the **comparison**. This package attacks each error at its statistical root. There are three roots.

## 2. False wins: dependence makes naive intervals too narrow

### The mechanism

Suppose you run a 198-item benchmark for 5 epochs (repeats) at temperature 0.7. You now have n = 990 graded observations, and the naive standard error of the mean is s/√990 — as if you had 990 independent measurements. You don't. The five answers to the same item share that item's difficulty: they are **positively correlated within item**. The same applies when items share a prompt template or task family.

For clustered data, the true variance of the mean score is inflated relative to the naive i.i.d. variance by the **design effect**:

```
DEFF = 1 + (K − 1)·ρ
```

where K is observations per cluster (epochs per item) and ρ is the intra-cluster correlation. The naive SE is therefore too small by a factor of √DEFF.

### Why that manufactures false wins

A hypothesis test is calibrated by its SE. If the SE is understated by factor c, your nominal z = 1.96 threshold actually corresponds to z = 1.96/c, and the real false-positive rate explodes:

| ρ (within-item correlation) | DEFF (K=5) | SE understated by | Real α at nominal 5% |
|---|---|---|---|
| 0.3 | 2.2 | 1.48× | ~19% |
| 0.5 | 3.0 | 1.73× | ~26% |
| 0.6 | 3.4 | 1.84× | ~29% |
| 1.0 (temp 0: identical epochs) | 5.0 | 2.24× | ~38% |

At realistic correlations, a team that believes it is running 5%-level tests is actually running ~25–30%-level tests. Roughly **one in four "significant" improvements is noise** — that is the false-win factory. The temperature-0 row is the limiting case that makes the logic vivid: identical epochs add *zero* information, yet the naive SE still shrinks by √5.

### What the package does

`standard_error()` detects the dependence structure (repeated items across epochs, or shared `cluster_id`s) and computes a **cluster-robust (CR1 sandwich) SE**: residuals are summed within clusters before squaring, so within-cluster correlation — whatever its value, no model of ρ required — is absorbed into the variance estimate. Degrees of freedom come from the number of clusters, not the number of observations. The output deliberately prints naive and clustered side by side with the inflation factor, because the corrective insight is the *gap* between the number you were quoting and the honest one. Equivalently, it reports the **effective sample size** n/DEFF: your 990 observations may be worth ~200.

## 3. Missed wins: unpaired comparisons throw away the covariance

### The mechanism

When both models answer the **same items**, the variance of the estimated gap is

```
Var(Â − B̂) = Var(Â) + Var(B̂) − 2·Cov(Â, B̂)
```

Eval items have heterogeneous difficulty, and difficulty is shared across models: a question that is hard for A tends to be hard for B. That makes Cov(Â, B̂) large and positive. The standard practice — compute each model's score ± SE independently and eyeball whether the intervals overlap — implicitly sets Cov = 0, leaving the comparison variance at its *maximum*.

The fix is **within-item differencing**: compute d_i = score_A(i) − score_B(i) per item and analyze the d_i directly. The item difficulty term cancels out of every d_i; only genuine model disagreement remains. For binary outcomes this becomes **McNemar's test**: items where both models are right or both wrong carry no comparative information at all — inference runs entirely on the discordant pairs.

### Worked numbers (GPQA Diamond shape: N = 198 items, accuracy ≈ 80%)

- Single-model SE: √(0.8·0.2/198) ≈ **2.8 points**.
- Unpaired difference SE: √2 × 2.8 ≈ **4.0 points**. A true 2–3 point improvement sits well inside ±1.96·SE — *invisible*. Verdict: "no significant difference." That is the missed win.
- Paired: suppose the models disagree on ~12% of items (typical when both are around 80% with shared difficulty). The per-item difference d_i is nonzero only on those items, giving Var(d_i) ≈ 0.12 and SE_d = √(0.12/198) ≈ **2.5 points** — a **~2.6× variance reduction** from the *same data, same spend*. A 3-point true gap is now near the detection boundary instead of buried, and a CI on the difference itself ([0.8, 5.2] rather than two overlapping marginal intervals) supports an actual decision.

Pairing costs nothing — the data is already paired whenever both models ran the same benchmark. Not exploiting it is pure statistical waste, and the missed wins it causes are silent: nothing alerts you to the improvement you discarded.

### What the package does

`compare()` pairs automatically when models share item IDs (paired-t on within-item differences for continuous scores; McNemar on discordant pairs for binary), and reports the **variance-reduction factor** versus the unpaired analysis so the value of the design is visible. When pairing is impossible it falls back to Welch / two-proportion tests — *with an explicit warning quantifying how much wider the interval became*, because knowing what you gave up is how eval designs improve next time.

## 4. Both errors at once: running evals without a power analysis

### The mechanism

The **minimum detectable effect** at significance α and power 1−β is

```
MDE ≈ (z_{1−α/2} + z_{1−β}) · SE_design ≈ 2.8 · SE_design   (α = 5%, power = 80%)
```

where SE_design is the standard error *of the comparison actually being run* — unpaired vs. paired, clustered vs. not. Concretely, near 50% accuracy on a 198-item benchmark, an unpaired comparison has SE_design ≈ 5.0 points, so the MDE is ≈ **14 points**. Most teams would guess a 198-item benchmark can "see" a 5-point improvement. It cannot, unpaired — not even close. Pairing (say 3× variance reduction) brings the MDE to ≈ 8 points; detecting a 3-point gap at 80% power requires pairing *plus* more items or epochs — a number you want **before** spending the inference budget.

### Why underpowering causes false wins too, not just missed wins

The obvious cost of low power is the missed win: a real 3-point improvement faces a coin-flip-or-worse chance of detection. The subtler cost is the **significance filter** (winner's curse): conditional on clearing the significance bar, an underpowered estimate is biased upward — at 20–30% power, "significant" effects are exaggerated ~2× or more (Gelman & Carlin's Type M error), and may even have the wrong sign (Type S). Teams that run many underpowered comparisons and ship whatever reaches significance are systematically shipping overestimated, sometimes spurious, improvements. Power analysis is therefore false-win protection as much as missed-win protection.

### What the package does

`power()` solves in both directions — required n for a target MDE, or detectable MDE at your current n — using the variance of the design you will actually run: the paired-difference variance when pairing applies (estimated from a pilot's observed discordance/covariance, not a worst-case guess) and inflated by the design effect when epochs/templates cluster. The output is the honest answer to "is this eval even capable of seeing the effect I care about?" — asked before the run, when it can still change the design.

## 5. The connective tissue: decision-first reporting

Each result object leads with the decision-relevant quantities — estimated gap, CI on the gap, effective sample size, what the design bought (variance reduction) or cost (inflation factor) — rather than a bare p-value. This is not cosmetic. False wins and missed wins are *decision* errors; surfacing "the honest interval is 2.2× wider than the one you were about to quote" or "pairing shrank your uncertainty 2.6× for free" at the moment of decision is what converts correct statistics into correct ship/no-ship calls.

## 6. Summary: failure mode → root cause → countermeasure

| Failure mode | Statistical root cause | Package countermeasure |
|---|---|---|
| False win | Naive SE ignores epoch/template dependence → real α is 25–30%, not 5% | Cluster-robust SE with inflation factor and effective n (`standard_error`) |
| Missed win | Unpaired comparison discards Cov from shared item difficulty → SE up to ~√2–3× too wide | Paired-t / McNemar with variance-reduction reporting (`compare`) |
| Missed win | Eval too small to detect the effect of interest | MDE / required-n planning under the true design (`power`) |
| False win | Significance filter on underpowered evals inflates "significant" effects (Type M/S) | Same — adequate power makes significant estimates honest |
| Both | p-value-first reporting obscures decision quantities | CI-on-the-gap, decision-first output with explicit design warnings |

## 7. What this package does *not* claim

Existing frameworks are not statistics-free: Inspect AI computes CLT standard errors per eval and offers bootstrap for non-mean statistics. The narrow, defensible gap is that they provide uncertainty on a *single* score under *independence*. Comparison between systems, correction when independence breaks, and power planning are the scope here — nothing more. The methods themselves (cluster-robust variance, paired designs, McNemar, power analysis) are standard measurement science; the contribution is wiring them into the eval workflow where the decisions actually happen.
