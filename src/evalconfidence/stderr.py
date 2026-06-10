"""Honest standard errors for a single eval score under dependence.

The naive CLT standard error treats every graded observation as independent.
The moment an eval runs multiple epochs per item, or items share a template /
task family, that assumption breaks and the naive interval is too narrow.
``standard_error`` reports the naive and cluster-robust numbers side by side,
plus the inflation factor, so the gap is visible rather than silent.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from scipy import stats

from .types import ItemResult, SEResult


def standard_error(
    results: Sequence[ItemResult],
    *,
    cluster: str | None = "auto",
    alpha: float = 0.05,
) -> SEResult:
    """Compute the SE of the mean score, naive and (when applicable) cluster-robust.

    Parameters
    ----------
    results:
        Scored items from a *single* system. Pass two systems to ``compare``
        instead; mixing them here raises.
    cluster:
        ``"auto"`` clusters by ``item_id`` when items repeat (epochs), else by
        ``cluster_id`` when populated, else not at all. ``"item"`` and
        ``"cluster_id"`` force a grouping; ``None`` disables clustering.
    alpha:
        Significance level for the reported confidence intervals.
    """
    if len(results) < 2:
        raise ValueError("Need at least 2 results to estimate a standard error.")
    models = {r.model_id for r in results}
    if len(models) > 1:
        raise ValueError(
            f"standard_error expects results from a single system, got {sorted(models)}. "
            "Filter to one model_id, or use compare() for two systems."
        )

    scores = np.asarray([r.score for r in results], dtype=float)
    n = scores.size
    mean = float(scores.mean())
    se_naive = float(scores.std(ddof=1) / np.sqrt(n))
    t_naive = stats.t.ppf(1 - alpha / 2, df=n - 1)
    ci_naive = (mean - t_naive * se_naive, mean + t_naive * se_naive)

    notes: list[str] = []
    cluster_by, labels = _resolve_cluster(results, cluster)
    if cluster_by is not None and labels is not None:
        unique = set(labels)
        if len(unique) < 2:
            notes.append(
                f"Only one cluster found by {cluster_by}; cluster-robust SE undefined, "
                "reporting naive only."
            )
            cluster_by = None
        elif len(unique) == n:
            notes.append(
                f"Every observation is its own cluster by {cluster_by}; "
                "clustering has no effect, reporting naive only."
            )
            cluster_by = None

    if cluster_by is None or labels is None:
        return SEResult(
            mean=mean, n=n, alpha=alpha,
            se_naive=se_naive, ci_naive=ci_naive, notes=notes,
        )

    se_cl, n_clusters = _cluster_robust_se(scores, labels)
    t_cl = stats.t.ppf(1 - alpha / 2, df=n_clusters - 1)
    ci_cl = (mean - t_cl * se_cl, mean + t_cl * se_cl)
    inflation = se_cl / se_naive
    if inflation < 1.0:
        notes.append(
            "Clustered SE came out below the naive SE (negative within-cluster "
            "correlation or noise); this can happen and is not an error."
        )
    return SEResult(
        mean=mean, n=n, alpha=alpha,
        se_naive=se_naive, ci_naive=ci_naive,
        cluster_by=cluster_by, n_clusters=n_clusters,
        se_clustered=se_cl, ci_clustered=ci_cl,
        inflation=inflation, design_effect=inflation**2,
        notes=notes,
    )


def _resolve_cluster(
    results: Sequence[ItemResult], cluster: str | None
) -> tuple[str | None, list[str] | None]:
    """Map the ``cluster`` argument to a grouping label per observation."""
    if cluster is None:
        return None, None
    if cluster == "auto":
        item_counts = Counter(r.item_id for r in results)
        if max(item_counts.values()) > 1:
            cluster = "item"
        elif any(r.cluster_id is not None for r in results):
            cluster = "cluster_id"
        else:
            return None, None
    if cluster == "item":
        return "item", [r.item_id for r in results]
    if cluster == "cluster_id":
        missing = sum(1 for r in results if r.cluster_id is None)
        if missing:
            raise ValueError(
                f"cluster='cluster_id' but {missing} of {len(results)} results "
                "have cluster_id=None."
            )
        return "cluster_id", [r.cluster_id for r in results]  # type: ignore[list-item]
    raise ValueError(f"Unknown cluster option: {cluster!r}. Use 'auto', 'item', 'cluster_id', or None.")


def _cluster_robust_se(scores: np.ndarray, labels: Sequence[str]) -> tuple[float, int]:
    """CR1 cluster-robust SE of the mean: sum residuals within clusters first.

    Var(mean) = G/(G-1) * sum_g S_g^2 / n^2, where S_g is the sum of residuals
    in cluster g. With temperature-0 epochs (identical repeats) this recovers
    the per-item variance exactly; with independent observations it converges
    to the naive variance.
    """
    codes = np.unique(np.asarray(labels), return_inverse=True)[1]
    n_clusters = int(codes.max()) + 1
    residuals = scores - scores.mean()
    cluster_sums = np.bincount(codes, weights=residuals)
    var = (n_clusters / (n_clusters - 1)) * float(np.sum(cluster_sums**2)) / scores.size**2
    return float(np.sqrt(var)), n_clusters
