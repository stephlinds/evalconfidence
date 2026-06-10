"""Core data types: the normalized input representation and result objects."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ItemResult:
    """One scored item from one system, the normalized unit every function consumes.

    A stable ``item_id`` shared across models/epochs is what enables pairing;
    ``epoch`` and ``cluster_id`` are what enable dependence-aware uncertainty.
    """

    item_id: str
    model_id: str
    score: float
    epoch: int = 0
    cluster_id: str | None = None
    correct: bool | None = None


@dataclass(frozen=True)
class SEResult:
    """Naive vs. dependence-aware standard error for a single eval score.

    ``se_clustered`` is None when no dependence structure was found (or
    clustering was explicitly disabled), in which case the naive number is
    the honest number.
    """

    mean: float
    n: int
    alpha: float
    se_naive: float
    ci_naive: tuple[float, float]
    cluster_by: str | None = None
    n_clusters: int | None = None
    se_clustered: float | None = None
    ci_clustered: tuple[float, float] | None = None
    inflation: float | None = None
    design_effect: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def se(self) -> float:
        """The honest SE: clustered when available, naive otherwise."""
        return self.se_clustered if self.se_clustered is not None else self.se_naive

    @property
    def ci(self) -> tuple[float, float]:
        """The honest CI: clustered when available, naive otherwise."""
        return self.ci_clustered if self.ci_clustered is not None else self.ci_naive

    def summary(self) -> str:
        conf = f"{100 * (1 - self.alpha):g}%"
        lines = [f"Mean score: {self.mean:.4f}  (n={self.n} observations)"]
        lines.append(
            f"  Naive i.i.d. SE:    {self.se_naive:.4f}  ->  {conf} CI "
            f"[{self.ci_naive[0]:.4f}, {self.ci_naive[1]:.4f}]"
        )
        if self.se_clustered is not None:
            lines.append(
                f"  Cluster-robust SE:  {self.se_clustered:.4f}  ->  {conf} CI "
                f"[{self.ci_clustered[0]:.4f}, {self.ci_clustered[1]:.4f}]"
                f"  ({self.n_clusters} clusters by {self.cluster_by})"
            )
            lines.append(
                f"  Inflation: {self.inflation:.2f}x  (design effect {self.design_effect:.2f})"
            )
            lines.append(
                f"  The naive interval treats all {self.n} observations as independent; "
                f"the cluster-robust interval accounts for dependence within the "
                f"{self.n_clusters} clusters and is the honest one to report."
            )
        else:
            lines.append("  No dependence structure detected; the naive SE is appropriate.")
        for note in self.notes:
            lines.append(f"  Note: {note}")
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - convenience passthrough
        return self.summary()
