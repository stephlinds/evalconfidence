# evalconfidence

**Decision-grade statistics for AI evals.** A companion layer — not another framework — that adds paired comparisons, dependence-aware uncertainty, and power analysis on top of the eval stack you already use (Inspect AI, or anything that can produce a dataframe).

> **Status: pre-release (v1 in progress).** `standard_error()` and the adapters work today; `compare()` and `power()` are landing next.

## The gap, stated honestly

Existing frameworks *do* quantify uncertainty: Inspect AI computes per-eval standard errors via the CLT and offers bootstrapping for non-mean statistics. What they give you is a defensible standard error on a **single** score under an **independence** assumption. What they don't give you:

- **Rigorous comparison between two systems** (paired tests that exploit shared items),
- **Correction when the independence assumption breaks** (repeated epochs, shared prompt templates),
- **Power / sample-size planning** before you spend the inference budget.

That's the whole scope of this package: results in, rigorous comparison out. No model calls, no orchestration, no tracing.

## How it works: the two-stage flow

This package never makes API calls — `model_id` is just a grouping label, never an endpoint. The flow has two stages, and the package only lives in the second:

1. **Generation (upstream, not this package).** An eval framework runs the model against the benchmark and grades outputs. This is where API calls, keys, and cost live. Inspect AI saves its own durable record automatically — a `.eval` log in `./logs/` with every prompt, response, and score per sample. A homegrown harness's CSV plays the same role.
2. **Analysis (this package).** An adapter reads that already-existing record into the normalized `ItemResult` rows — `from_inspect()` for `.eval` logs, `from_dataframe()` for anything tabular — and the statistics functions compute on those fixed numbers. No model is ever consulted again.

This separation is what makes analyses cheaply reproducible: pay for stage 1 once, keep the log/CSV, and re-run stage 2 forever for free.

**What gets saved:** stage-1 artifacts are saved by whoever produced them (Inspect does this automatically). Stage-2 outputs are returned as in-memory dataclasses (`SEResult`, ...) — print them or serialize with `dataclasses.asdict()`; the package deliberately doesn't persist analysis results, because the saved stage-1 record is the thing worth keeping and the statistics re-run in milliseconds.

## Quick example

```python
from evalconfidence import from_inspect, standard_error

results = from_inspect("logs/2026-06-10T12-00-00_gpqa_diamond.eval")
print(standard_error(results))  # 198 items x 5 epochs
```

```
Mean score: 0.8424  (n=990 observations)
  Naive i.i.d. SE:    0.0116  ->  95% CI [0.8197, 0.8651]
  Cluster-robust SE:  0.0252  ->  95% CI [0.7926, 0.8922]  (198 clusters by item)
  Inflation: 2.18x  (design effect 4.74)
  The naive interval treats all 990 observations as independent; the cluster-robust
  interval accounts for dependence within the 198 clusters and is the honest one to report.
```

The naive SE treated 990 observations as independent and shrank by ~√5; clustering by item recovers the true ~198-item precision. Teams quoting the smaller number call differences significant that aren't.

Not on Inspect? Use the escape hatch:

```python
from evalconfidence import from_dataframe
results = from_dataframe(df, item_id="qid", model_id="system", score="acc")
```

## Install

```bash
pip install -e .            # core: numpy + scipy only
pip install -e ".[inspect]" # + Inspect AI log reading
pip install -e ".[dev]"     # + pytest, pandas (for tests)
```

## Roadmap (v1)

- [x] `ItemResult` normalized representation + `from_inspect` + `from_dataframe`
- [x] `standard_error()` — naive vs. cluster-robust side by side, inflation factor
- [ ] `compare()` — paired comparison of two systems (paired-t / McNemar), variance-reduction factor, unpaired fallback with warning
- [ ] `power()` — required n ↔ minimum detectable effect, pairing- and cluster-aware
- [ ] Demo notebook on GPQA Diamond (reproducible for under $5)

License: Apache-2.0
