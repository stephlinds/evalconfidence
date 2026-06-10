"""Adapters that populate the normalized ItemResult representation.

``from_inspect`` parses Inspect AI eval logs; ``from_dataframe`` is the escape
hatch that makes the package work with any framework's output via a tidy frame.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from .types import ItemResult

# Inspect's categorical score values, per inspect_ai.scorer conventions.
_VALUE_MAP = {"C": 1.0, "I": 0.0, "P": 0.5, "N": 0.0}


def from_dataframe(
    df: Any,
    *,
    item_id: str = "item_id",
    model_id: str = "model_id",
    score: str | None = "score",
    epoch: str | None = "epoch",
    cluster_id: str | None = "cluster_id",
    correct: str | None = "correct",
) -> list[ItemResult]:
    """Build ItemResults from a tidy dataframe; keyword args remap column names.

    ``epoch``, ``cluster_id``, and ``correct`` are optional: pass ``None`` or
    leave the column absent and defaults are used. If ``score`` is absent but
    ``correct`` is present, scores are derived as 0/1 from ``correct``.
    """
    cols = set(df.columns)
    for required, name in ((item_id, "item_id"), (model_id, "model_id")):
        if required not in cols:
            raise ValueError(f"Column {required!r} (for {name}) not found in dataframe.")

    score_col = score if score is not None and score in cols else None
    correct_col = correct if correct is not None and correct in cols else None
    if score_col is None and correct_col is None:
        raise ValueError(
            f"Need a score column ({score!r}) or a correct column ({correct!r})."
        )
    epoch_col = epoch if epoch is not None and epoch in cols else None
    cluster_col = cluster_id if cluster_id is not None and cluster_id in cols else None

    out: list[ItemResult] = []
    for row in df.itertuples(index=False):
        get = lambda col: getattr(row, col)
        correct_val = bool(get(correct_col)) if correct_col is not None else None
        score_val = float(get(score_col)) if score_col is not None else float(correct_val)
        cluster_val = get(cluster_col) if cluster_col is not None else None
        out.append(
            ItemResult(
                item_id=str(get(item_id)),
                model_id=str(get(model_id)),
                score=score_val,
                epoch=int(get(epoch_col)) if epoch_col is not None else 0,
                cluster_id=None if cluster_val is None else str(cluster_val),
                correct=correct_val,
            )
        )
    return out


def from_inspect(
    log: Any,
    *,
    scorer: str | None = None,
    cluster_field: str | None = None,
) -> list[ItemResult]:
    """Build ItemResults from an Inspect AI eval log (path or EvalLog object).

    ``scorer`` selects among multiple scorers (required only when ambiguous).
    ``cluster_field`` names a sample metadata key to use as ``cluster_id``
    (e.g. a subject/domain label).
    """
    if isinstance(log, (str, os.PathLike)):
        try:
            from inspect_ai.log import read_eval_log
        except ImportError as exc:  # pragma: no cover - exercised only without inspect-ai
            raise ImportError(
                "Reading a log file requires inspect-ai. Install with: "
                "pip install 'evalconfidence[inspect]'"
            ) from exc
        log = read_eval_log(str(log))

    samples = getattr(log, "samples", None)
    if not samples:
        raise ValueError(
            "Eval log has no samples (was it read header-only?). "
            "Re-read with header_only=False."
        )
    model = str(getattr(log.eval, "model", "unknown"))

    out: list[ItemResult] = []
    for sample in samples:
        scores = getattr(sample, "scores", None) or {}
        if scorer is not None:
            if scorer not in scores:
                raise ValueError(
                    f"Scorer {scorer!r} not in sample scores {sorted(scores)}."
                )
            value = scores[scorer].value
        elif len(scores) == 1:
            value = next(iter(scores.values())).value
        else:
            raise ValueError(
                f"Sample has {len(scores)} scorers {sorted(scores)}; "
                "pass scorer=... to pick one."
            )
        score_val, correct_val = _value_to_score(value)
        metadata = getattr(sample, "metadata", None) or {}
        cluster_val = metadata.get(cluster_field) if cluster_field else None
        out.append(
            ItemResult(
                item_id=str(sample.id),
                model_id=model,
                score=score_val,
                epoch=int(getattr(sample, "epoch", 0) or 0),
                cluster_id=None if cluster_val is None else str(cluster_val),
                correct=correct_val,
            )
        )
    return out


def _value_to_score(value: Any) -> tuple[float, bool | None]:
    """Normalize an Inspect score value to (float score, optional correctness)."""
    if isinstance(value, str):
        if value not in _VALUE_MAP:
            raise ValueError(f"Unrecognized categorical score value {value!r}.")
        # Partial credit is neither correct nor incorrect.
        return _VALUE_MAP[value], None if value == "P" else value == "C"
    if isinstance(value, bool):
        return float(value), value
    if isinstance(value, (int, float)):
        score = float(value)
        return score, (score == 1.0) if score in (0.0, 1.0) else None
    raise ValueError(f"Cannot convert score value {value!r} to float.")
