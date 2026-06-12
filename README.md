# evalconfidence

**Decision-grade statistics for AI evals.** A companion layer — not another framework — that adds paired comparisons, dependence-aware uncertainty, and power analysis on top of the eval stack you already use (Inspect AI, or anything that can produce a dataframe).

> **Status: pre-release (v1 in progress).** The full statistics layer works today: `standard_error()`, `compare()`, `power()`, and the adapters, plus a demo notebook ([examples/demo.ipynb](examples/demo.ipynb)) that runs with zero API keys, and CI on Python 3.10/3.12. Remaining for v1: re-render the notebook on real GPQA Diamond logs.

## The gap, stated honestly

Existing frameworks *do* quantify uncertainty: Inspect AI computes per-eval standard errors via the CLT, offers bootstrapping for non-mean statistics, and — since v0.3.64 (Feb 2025) — supports clustered standard errors via `stderr(cluster=...)` when you declare a grouping field. What they give you is a defensible standard error on a **single** score. What none of them give you (checked against the Inspect changelog and DeepEval metrics list, June 2026):

- **Rigorous comparison between two systems** — paired tests that exploit shared items, a CI on the *difference*, McNemar for binary scores. The universal practice is still eyeballing two separate intervals, which is an unpaired test at its maximum variance.
- **Power / sample-size planning** before you spend the inference budget — how many items to detect the gap you care about, or the smallest gap your benchmark can see at all.

On dependence-aware uncertainty the gap is narrower and we say so: Inspect can cluster if you name the grouping up front. This package adds the **diagnostic** framing — naive and cluster-robust side by side with the inflation factor, epoch structure auto-detected — and works on results from any framework, not just Inspect tasks configured with custom metrics.

That's the whole scope of this package: results in, rigorous comparison out. No model calls, no orchestration, no tracing.

### Capability matrix

| Capability | Existing frameworks | evalconfidence |
|---|---|---|
| Run / orchestrate / trace / score evals | Yes | No (consumes results) |
| Single-score standard error | Yes (Inspect: CLT, bootstrap) | Re-derives, reported side by side |
| Clustered standard errors | Partial (Inspect `stderr(cluster=...)`, declared field) | **Yes — auto-detected epochs, inflation factor, any framework** |
| **Paired comparison of two systems** | No | **Yes — paired-t / McNemar, CI on the difference** |
| **Power / minimum detectable effect** | No | **Yes — n ↔ MDE, pairing- and cluster-aware** |
| Judge debiasing (PPI) | No | Planned (v2) |

For the full technical argument — how dependence-blind SEs manufacture false wins at a real α of ~25–30%, how unpaired comparisons silently bury real improvements, and why underpowered evals cause *both* errors — see [docs/why-it-works.md](docs/why-it-works.md).

## How it works: the two-stage flow

This package never makes API calls — `model_id` is just a grouping label, never an endpoint. The flow has two stages, and the package only lives in the second:

1. **Generation (upstream, not this package).** An eval framework runs the model against the benchmark and grades outputs. This is where API calls, keys, and cost live. Inspect AI saves its own durable record automatically — a `.eval` log in `./logs/` with every prompt, response, and score per sample. A homegrown harness's CSV plays the same role.
2. **Analysis (this package).** An adapter reads that already-existing record into the normalized `ItemResult` rows — `from_inspect()` for `.eval` logs, `from_dataframe()` for anything tabular — and the statistics functions compute on those fixed numbers. No model is ever consulted again.

This separation is what makes analyses cheaply reproducible: pay for stage 1 once, keep the log/CSV, and re-run stage 2 forever for free.

**What gets saved:** stage-1 artifacts are saved by whoever produced them (Inspect does this automatically). Stage-2 outputs are returned as in-memory dataclasses (`SEResult`, ...) — print them or serialize with `dataclasses.asdict()`; the package deliberately doesn't persist analysis results, because the saved stage-1 record is the thing worth keeping and the statistics re-run in milliseconds.

## Quick example

```python
from evalconfidence import from_inspect, compare, power, standard_error

results_a = from_inspect("logs/..._gpqa_diamond_model-a.eval")  # 198 items x 5 epochs
results_b = from_inspect("logs/..._gpqa_diamond_model-b.eval")

print(compare(results_a, results_b))          # pairs on shared items automatically
print(standard_error(results_a))              # naive vs cluster-robust, side by side
print(power((results_a, results_b), mde=0.03))  # items needed to detect 3 points
```

Output (from [the demo notebook](examples/demo.ipynb), where the true gap is known to be 4.5 points):

```
model-a is estimated to outperform model-b by 4.3 points, 95% CI [1.0, 7.7] (A−B).
The difference is significant at alpha=0.05 (p=0.0122, paired_t).
Pairing reduced the comparison variance by 4.5x: the 198 paired items deliver
the precision of ~882 unpaired items.

Mean score: 0.5576  (n=990 observations)
  Naive i.i.d. SE:    0.0158  ->  95% CI [0.5266, 0.5886]
  Cluster-robust SE:  0.0253  ->  95% CI [0.5076, 0.6075]  (198 clusters by item)
  Inflation: 1.60x  (design effect 2.57)

Detecting a 3.0 points gap at alpha=0.05 with 80% power requires ~510 paired items.
```

The same data, compared unpaired (the eyeball-the-two-intervals test), give 95% CI [−2.8, +11.5], p = 0.23 — a real 4.5-point improvement written off as noise. The full story, with figures, is in the [demo notebook](examples/demo.ipynb).

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
pip install -e ".[demo]"    # + matplotlib, jupyter (for the demo notebook)
```

## Roadmap (v1)

- [x] `ItemResult` normalized representation + `from_inspect` + `from_dataframe`
- [x] `standard_error()` — naive vs. cluster-robust side by side, inflation factor
- [x] `compare()` — paired comparison of two systems (paired-t / McNemar), variance-reduction factor, unpaired fallback with warning
- [x] `power()` — required n ↔ minimum detectable effect, pairing- and cluster-aware
- [x] Demo notebook — three figures (wrong winner / false confidence / budget planning) on a synthetic GPQA-shaped DGP, no API keys needed: [examples/demo.ipynb](examples/demo.ipynb)
- [ ] Re-render the notebook on real GPQA Diamond logs (one generation run, reproducible for under $5)

License: Apache-2.0
