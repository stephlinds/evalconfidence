"""Paired comparison of two systems — the flagship function.

When both systems answered the same items, the variance of the estimated gap
is Var(A) + Var(B) - 2*Cov(A, B), and shared item difficulty makes Cov large.
Within-item differencing cancels it; ignoring it (the independent-intervals
eyeball test) leaves the comparison variance at its maximum. See
docs/why-it-works.md for the full argument.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .types import ItemResult

# Spec §10 decision 4 (provisional): auto-pair when at least this fraction of
# the larger item set is shared between the two systems.
PAIRING_COVERAGE_THRESHOLD = 0.9

# At or below this many discordant pairs, McNemar uses the exact binomial
# p-value instead of the continuity-corrected chi-square approximation.
MCNEMAR_EXACT_MAX_DISCORDANT = 25


@dataclass(frozen=True)
class ComparisonResult:
    """Outcome of comparing system A against system B. ``diff`` is A minus B."""

    model_a: str
    model_b: str
    method: str  # "paired_t" | "mcnemar" | "welch_t" | "two_proportion"
    paired: bool
    n_items_a: int
    n_items_b: int
    n_pairs: int | None
    mean_a: float
    mean_b: float
    diff: float
    ci_diff: tuple[float, float]
    p_value: float
    alpha: float
    variance_reduction: float | None = None
    effective_n: float | None = None
    odds_ratio: float | None = None
    ci_odds_ratio: tuple[float, float] | None = None
    n_discordant: int | None = None
    scores_are_unit_scale: bool = True
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    def _fmt(self, x: float) -> str:
        return f"{100 * x:.1f}" if self.scores_are_unit_scale else f"{x:.3f}"

    def summary(self) -> str:
        unit = " points" if self.scores_are_unit_scale else ""
        winner, loser, gap = (
            (self.model_a, self.model_b, self.diff)
            if self.diff >= 0
            else (self.model_b, self.model_a, -self.diff)
        )
        lo, hi = self.ci_diff
        conf = f"{100 * (1 - self.alpha):g}%"
        lines = [
            f"{winner} is estimated to outperform {loser} by {self._fmt(gap)}{unit}, "
            f"{conf} CI [{self._fmt(lo)}, {self._fmt(hi)}] (A−B). The difference is "
            f"{'significant' if self.significant else 'NOT significant'} at "
            f"alpha={self.alpha:g} (p={self.p_value:.4f}, {self.method})."
        ]
        if self.paired and self.variance_reduction is not None:
            lines.append(
                f"Pairing reduced the comparison variance by "
                f"{self.variance_reduction:.1f}x: the {self.n_pairs} paired items "
                f"deliver the precision of ~{self.effective_n:.0f} unpaired items."
            )
        if self.method == "mcnemar":
            lines.append(
                f"Models disagree on {self.n_discordant} of {self.n_pairs} items; "
                f"discordant-pair odds ratio {self.odds_ratio:.2f}, "
                f"{conf} CI [{self.ci_odds_ratio[0]:.2f}, {self.ci_odds_ratio[1]:.2f}]."
            )
        lines.extend(f"WARNING: {w}" for w in self.warnings)
        lines.extend(f"Note: {n}" for n in self.notes)
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - convenience passthrough
        return self.summary()


def compare(
    results_a: Sequence[ItemResult],
    results_b: Sequence[ItemResult],
    *,
    paired: bool | str = "auto",
    alpha: float = 0.05,
) -> ComparisonResult:
    """Compare two systems' scored results; pair on shared item_ids when possible.

    Epoch repeats are aggregated to per-item means first, so every test runs on
    independent item-level observations (this also sidesteps within-item
    dependence). Binary single-shot results pair via McNemar; everything else
    paired uses the t-test on within-item differences. Unpaired falls back to
    Welch's t (continuous) or a two-proportion z with a Newcombe CI (binary) —
    with an explicit warning, because the widened interval is part of the story.
    """
    a = _ItemSummary.build(results_a, "results_a")
    b = _ItemSummary.build(results_b, "results_b")
    notes: list[str] = []
    warnings: list[str] = []
    if a.model == b.model:
        notes.append(
            f"Both sides carry model_id={a.model!r}; assuming they are two distinct "
            "systems that happen to share a label."
        )
    if a.epochs_aggregated or b.epochs_aggregated:
        notes.append("Epoch repeats were aggregated to per-item mean scores.")

    shared = sorted(a.means.keys() & b.means.keys())
    coverage = len(shared) / max(len(a.means), len(b.means))
    if paired == "auto":
        do_pair = coverage >= PAIRING_COVERAGE_THRESHOLD and len(shared) >= 2
        if not do_pair:
            warnings.append(
                f"Pairing unavailable: only {coverage:.0%} of items are shared "
                f"(threshold {PAIRING_COVERAGE_THRESHOLD:.0%}). Falling back to an "
                "unpaired comparison, which ignores the covariance from shared item "
                "difficulty and therefore widens the interval. Re-running both "
                "systems on identical items would tighten this comparison for free."
            )
    elif paired is True:
        if len(shared) < 2:
            raise ValueError(
                f"paired=True but only {len(shared)} shared item_id(s) between the "
                "two result sets."
            )
        do_pair = True
    elif paired is False:
        do_pair = False
        if len(shared) >= 2:
            warnings.append(
                f"paired=False forced an unpaired comparison although {len(shared)} "
                "items are shared; the interval is wider than it needs to be."
            )
    else:
        raise ValueError(f"paired must be True, False, or 'auto', got {paired!r}.")

    if do_pair and coverage < 1.0:
        dropped = max(len(a.means), len(b.means)) - len(shared)
        notes.append(f"{dropped} unshared item(s) dropped from the paired analysis.")

    unit_scale = a.unit_scale and b.unit_scale
    common = dict(
        model_a=a.model, model_b=b.model,
        n_items_a=len(a.means), n_items_b=len(b.means),
        alpha=alpha, scores_are_unit_scale=unit_scale,
        warnings=warnings, notes=notes,
    )
    if do_pair:
        xa = np.array([a.means[i] for i in shared])
        xb = np.array([b.means[i] for i in shared])
        if a.binary_single_shot and b.binary_single_shot:
            return _mcnemar(xa, xb, common=common)
        return _paired_t(xa, xb, common=common)
    if a.binary_single_shot and b.binary_single_shot:
        return _two_proportion(a.values(), b.values(), common=common)
    return _welch_t(a.values(), b.values(), common=common)


class _ItemSummary:
    """Per-item mean scores for one system, plus shape flags."""

    def __init__(self, model, means, epochs_aggregated, binary_single_shot, unit_scale):
        self.model = model
        self.means = means
        self.epochs_aggregated = epochs_aggregated
        self.binary_single_shot = binary_single_shot
        self.unit_scale = unit_scale

    def values(self) -> np.ndarray:
        return np.array(list(self.means.values()))

    @staticmethod
    def build(results: Sequence[ItemResult], arg_name: str) -> "_ItemSummary":
        if len(results) < 2:
            raise ValueError(f"{arg_name} needs at least 2 results.")
        models = {r.model_id for r in results}
        if len(models) > 1:
            raise ValueError(
                f"{arg_name} mixes models {sorted(models)}; pass one system per argument."
            )
        by_item: dict[str, list[float]] = defaultdict(list)
        for r in results:
            by_item[r.item_id].append(r.score)
        scores = np.array([r.score for r in results])
        single_shot = all(len(v) == 1 for v in by_item.values())
        return _ItemSummary(
            model=models.pop(),
            means={k: float(np.mean(v)) for k, v in by_item.items()},
            epochs_aggregated=not single_shot,
            binary_single_shot=single_shot and bool(np.isin(scores, (0.0, 1.0)).all()),
            unit_scale=bool((scores >= 0.0).all() and (scores <= 1.0).all()),
        )


def _variance_reduction(xa, xb, var_paired_mean):
    """Unpaired-vs-paired variance ratio of the estimated gap, on the same items."""
    n = xa.size
    var_unpaired_mean = (np.var(xa, ddof=1) + np.var(xb, ddof=1)) / n
    if var_paired_mean <= 0:
        return None, None
    vr = float(var_unpaired_mean / var_paired_mean)
    return vr, float(n * vr)


def _paired_t(xa, xb, *, common) -> ComparisonResult:
    d = xa - xb
    n = d.size
    mean_d = float(d.mean())
    sd = float(d.std(ddof=1))
    if sd == 0.0:
        common["notes"].append(
            "All within-item differences are identical; the paired SE is 0 and the "
            "p-value is degenerate."
        )
        return ComparisonResult(
            method="paired_t", paired=True, n_pairs=n,
            mean_a=float(xa.mean()), mean_b=float(xb.mean()),
            diff=mean_d, ci_diff=(mean_d, mean_d),
            p_value=0.0 if mean_d != 0 else 1.0, **common,
        )
    se = sd / np.sqrt(n)
    t_stat = mean_d / se
    p = 2 * stats.t.sf(abs(t_stat), df=n - 1)
    t_crit = stats.t.ppf(1 - common["alpha"] / 2, df=n - 1)
    vr, eff_n = _variance_reduction(xa, xb, se**2)
    return ComparisonResult(
        method="paired_t", paired=True, n_pairs=n,
        mean_a=float(xa.mean()), mean_b=float(xb.mean()),
        diff=mean_d, ci_diff=(mean_d - t_crit * se, mean_d + t_crit * se),
        p_value=float(p), variance_reduction=vr, effective_n=eff_n, **common,
    )


def _mcnemar(xa, xb, *, common) -> ComparisonResult:
    n = xa.size
    n10 = int(np.sum((xa == 1) & (xb == 0)))  # A right, B wrong
    n01 = int(np.sum((xa == 0) & (xb == 1)))  # A wrong, B right
    n_disc = n10 + n01
    diff = (n10 - n01) / n

    if n_disc == 0:
        common["notes"].append("Models agree on every item; no discordant pairs to test.")
        return ComparisonResult(
            method="mcnemar", paired=True, n_pairs=n, n_discordant=0,
            mean_a=float(xa.mean()), mean_b=float(xb.mean()),
            diff=0.0, ci_diff=(0.0, 0.0), p_value=1.0,
            odds_ratio=1.0, ci_odds_ratio=(0.0, np.inf), **common,
        )
    if n_disc <= MCNEMAR_EXACT_MAX_DISCORDANT:
        p = float(stats.binomtest(n10, n_disc, 0.5).pvalue)
        common["notes"].append(
            f"Exact binomial McNemar used ({n_disc} discordant pairs <= "
            f"{MCNEMAR_EXACT_MAX_DISCORDANT})."
        )
    else:
        chi2 = (abs(n10 - n01) - 1) ** 2 / n_disc
        p = float(stats.chi2.sf(chi2, df=1))

    # Haldane-Anscombe correction keeps the odds ratio finite when a cell is 0.
    c10, c01 = (n10 + 0.5, n01 + 0.5) if (n10 == 0 or n01 == 0) else (n10, n01)
    or_ = c10 / c01
    z = stats.norm.ppf(1 - common["alpha"] / 2)
    log_se = np.sqrt(1 / c10 + 1 / c01)
    ci_or = (float(or_ * np.exp(-z * log_se)), float(or_ * np.exp(z * log_se)))

    # CLT CI on the score difference itself (the decision-relevant scale).
    var_d = (n10 / n + n01 / n - diff**2) / n
    se_d = float(np.sqrt(var_d))
    ci = (diff - z * se_d, diff + z * se_d)
    vr, eff_n = _variance_reduction(xa, xb, var_d)
    return ComparisonResult(
        method="mcnemar", paired=True, n_pairs=n, n_discordant=n_disc,
        mean_a=float(xa.mean()), mean_b=float(xb.mean()),
        diff=float(diff), ci_diff=ci, p_value=p,
        odds_ratio=float(or_), ci_odds_ratio=ci_or,
        variance_reduction=vr, effective_n=eff_n, **common,
    )


def _welch_t(xa, xb, *, common) -> ComparisonResult:
    na, nb = xa.size, xb.size
    va, vb = np.var(xa, ddof=1), np.var(xb, ddof=1)
    se = float(np.sqrt(va / na + vb / nb))
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    )
    diff = float(xa.mean() - xb.mean())
    t_stat = diff / se
    p = 2 * stats.t.sf(abs(t_stat), df=df)
    t_crit = stats.t.ppf(1 - common["alpha"] / 2, df=df)
    return ComparisonResult(
        method="welch_t", paired=False, n_pairs=None,
        mean_a=float(xa.mean()), mean_b=float(xb.mean()),
        diff=diff, ci_diff=(diff - t_crit * se, diff + t_crit * se),
        p_value=float(p), **common,
    )


def _two_proportion(xa, xb, *, common) -> ComparisonResult:
    na, nb = xa.size, xb.size
    pa, pb = float(xa.mean()), float(xb.mean())
    pooled = (xa.sum() + xb.sum()) / (na + nb)
    se_pooled = np.sqrt(pooled * (1 - pooled) * (1 / na + 1 / nb))
    diff = pa - pb
    z_stat = diff / se_pooled if se_pooled > 0 else np.inf * np.sign(diff)
    p = float(2 * stats.norm.sf(abs(z_stat)))
    z = stats.norm.ppf(1 - common["alpha"] / 2)
    la, ua = _wilson(pa, na, z)
    lb, ub = _wilson(pb, nb, z)
    # Newcombe's score-interval difference (method 10).
    ci = (
        float(diff - np.sqrt((pa - la) ** 2 + (ub - pb) ** 2)),
        float(diff + np.sqrt((ua - pa) ** 2 + (pb - lb) ** 2)),
    )
    return ComparisonResult(
        method="two_proportion", paired=False, n_pairs=None,
        mean_a=pa, mean_b=pb, diff=float(diff), ci_diff=ci, p_value=p, **common,
    )


def _wilson(p: float, n: int, z: float) -> tuple[float, float]:
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return center - half, center + half
