"""Power and sample-size planning for eval comparisons — the tool nobody ships.

Answers, before the inference budget is spent: how many items does this eval
need to detect the effect I care about (``mde`` -> ``n``), or what is the
smallest effect this eval can see (``n`` -> ``mde``)? With both given, returns
the achieved power. The variance comes from the design actually being run:
the within-item difference variance when the pilot is paired, twice the
per-item variance for a two-arm unpaired comparison, inflated by the estimated
design effect when items cluster (shared templates / task families).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .compare import _ItemSummary
from .stderr import _cluster_robust_se
from .types import ItemResult


@dataclass(frozen=True)
class PowerResult:
    """One of ``n_required`` / ``mde_detectable`` / ``power_achieved`` is set,
    depending on which inputs were provided."""

    design: str  # "paired" | "unpaired" | "specified"
    variance: float  # per-item variance of the comparison quantity (post-DEFF)
    alpha: float
    target_power: float
    n_required: int | None = None
    mde_detectable: float | None = None
    power_achieved: float | None = None
    mde_input: float | None = None
    n_input: int | None = None
    design_effect: float | None = None
    scores_are_unit_scale: bool = False
    notes: list[str] = field(default_factory=list)

    def _fmt(self, x: float) -> str:
        return f"{100 * x:.1f} points" if self.scores_are_unit_scale else f"{x:.4g}"

    def summary(self) -> str:
        pct = f"{100 * self.target_power:g}%"
        if self.n_required is not None:
            head = (
                f"Detecting a {self._fmt(self.mde_input)} gap at alpha={self.alpha:g} "
                f"with {pct} power requires ~{self.n_required} {self.design} items."
            )
        elif self.mde_detectable is not None:
            head = (
                f"With {self.n_input} {self.design} items, the minimum detectable gap "
                f"at alpha={self.alpha:g} and {pct} power is {self._fmt(self.mde_detectable)}."
            )
        else:
            head = (
                f"With {self.n_input} {self.design} items, a true gap of "
                f"{self._fmt(self.mde_input)} is detected with "
                f"{100 * self.power_achieved:.0f}% power at alpha={self.alpha:g} "
                f"(target was {pct})."
            )
        lines = [head, f"Per-item comparison variance: {self.variance:.4g}."]
        if self.design_effect is not None:
            lines.append(
                f"Includes a {self.design_effect:.2f}x design-effect inflation from "
                "item clustering."
            )
        lines.extend(f"Note: {n}" for n in self.notes)
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - convenience passthrough
        return self.summary()


def power(
    results: Sequence[ItemResult] | tuple[Sequence[ItemResult], Sequence[ItemResult]] | None = None,
    *,
    mde: float | None = None,
    n: int | None = None,
    baseline_var: float | None = None,
    alpha: float = 0.05,
    power: float = 0.8,
) -> PowerResult:
    """Plan an eval comparison in either direction (normal approximation).

    Variance source (exactly one):
    - ``results`` as a pair ``(results_a, results_b)``: a paired pilot; uses the
      within-item difference variance Var(d_i) on shared items.
    - ``results`` as a single sequence: a one-system pilot; plans a two-arm
      unpaired comparison with ``2 * Var(item mean)`` (both arms assumed alike,
      ``n`` counts items per arm). For binary unpaired planning without a pilot,
      pass ``baseline_var = 2 * p * (1 - p)``.
    - ``baseline_var``: the per-item variance of the comparison quantity, used
      as-is.

    Question (via ``mde`` / ``n``): ``mde`` only -> required n; ``n`` only ->
    detectable MDE at ``power``; both -> achieved power.

    Epoch repeats in a pilot are aggregated to item means first; if the pilot
    carries ``cluster_id``s, the variance is inflated by the estimated design
    effect so the answer is honest for the clustered design.
    """
    if (results is None) == (baseline_var is None):
        raise ValueError("Provide exactly one variance source: results or baseline_var.")
    if mde is None and n is None:
        raise ValueError("Provide mde (-> required n), n (-> detectable MDE), or both (-> achieved power).")
    if mde is not None and mde <= 0:
        raise ValueError("mde must be positive.")
    if n is not None and n < 2:
        raise ValueError("n must be at least 2.")

    notes: list[str] = []
    deff = None
    unit_scale = False
    if baseline_var is not None:
        if baseline_var <= 0:
            raise ValueError("baseline_var must be positive.")
        design, variance = "specified", float(baseline_var)
    elif isinstance(results, tuple):
        design, variance, deff, unit_scale = _paired_pilot_variance(*results, notes=notes)
    else:
        design, variance, deff, unit_scale = _single_pilot_variance(results, notes=notes)

    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    common = dict(
        design=design, variance=variance, alpha=alpha, target_power=power,
        design_effect=deff, scores_are_unit_scale=unit_scale, notes=notes,
        mde_input=mde, n_input=n,
    )
    if mde is not None and n is not None:
        achieved = float(stats.norm.cdf(mde / math.sqrt(variance / n) - z_a))
        return PowerResult(power_achieved=achieved, **common)
    if mde is not None:
        required = math.ceil((z_a + z_b) ** 2 * variance / mde**2)
        return PowerResult(n_required=required, **common)
    detectable = float((z_a + z_b) * math.sqrt(variance / n))
    return PowerResult(mde_detectable=detectable, **common)


def _item_clusters(results: Sequence[ItemResult]) -> dict[str, str | None]:
    clusters: dict[str, str | None] = {}
    for r in results:
        if clusters.get(r.item_id) is None:
            clusters[r.item_id] = r.cluster_id
    return clusters


def _apply_design_effect(values: np.ndarray, cluster_labels, notes: list[str]):
    """Estimate DEFF = clustered/naive variance of the mean over item clusters."""
    labeled = [c is not None for c in cluster_labels]
    if not any(labeled):
        return None
    if not all(labeled):
        notes.append("Some items lack cluster_id; clustering ignored for power planning.")
        return None
    unique = set(cluster_labels)
    if len(unique) < 2 or len(unique) == len(values):
        return None
    se_cl, _ = _cluster_robust_se(values, list(cluster_labels))
    se_naive = values.std(ddof=1) / np.sqrt(values.size)
    if se_naive == 0:
        return None
    return float((se_cl / se_naive) ** 2)


def _paired_pilot_variance(results_a, results_b, *, notes):
    a = _ItemSummary.build(results_a, "results_a")
    b = _ItemSummary.build(results_b, "results_b")
    shared = sorted(a.means.keys() & b.means.keys())
    if len(shared) < 2:
        raise ValueError(
            f"Paired pilot needs >= 2 shared item_ids, found {len(shared)}."
        )
    if a.epochs_aggregated or b.epochs_aggregated:
        notes.append("Epoch repeats were aggregated to per-item mean scores.")
    d = np.array([a.means[i] - b.means[i] for i in shared])
    variance = float(d.var(ddof=1))
    if variance == 0:
        raise ValueError("Pilot difference variance is 0; cannot plan from it.")
    clusters = _item_clusters(list(results_a) + list(results_b))
    deff = _apply_design_effect(d, [clusters.get(i) for i in shared], notes)
    if deff is not None:
        variance *= deff
    notes.append(
        f"Variance estimated from a paired pilot of {len(shared)} shared items."
    )
    return "paired", variance, deff, a.unit_scale and b.unit_scale


def _single_pilot_variance(results, *, notes):
    s = _ItemSummary.build(results, "results")
    if s.epochs_aggregated:
        notes.append("Epoch repeats were aggregated to per-item mean scores.")
    values = s.values()
    item_var = float(values.var(ddof=1))
    if item_var == 0:
        raise ValueError("Pilot per-item variance is 0; cannot plan from it.")
    clusters = _item_clusters(results)
    deff = _apply_design_effect(
        values, [clusters.get(i) for i in s.means], notes
    )
    if deff is not None:
        item_var *= deff
    notes.append(
        f"Variance estimated from a single-system pilot of {len(s.means)} items; "
        "planning a two-arm unpaired comparison (n = items per arm, variance "
        "doubled for the second arm). A paired design would need fewer."
    )
    return "unpaired", 2 * item_var, deff, s.unit_scale


__all__ = ["power", "PowerResult"]
