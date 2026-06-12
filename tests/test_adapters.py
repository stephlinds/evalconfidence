"""Adapter tests: dataframe round-trip and a duck-typed fake Inspect log."""

from types import SimpleNamespace

import pandas as pd
import pytest

from evalconfidence import from_dataframe, from_inspect


class TestFromDataframe:
    def test_full_roundtrip(self):
        df = pd.DataFrame(
            {
                "item_id": ["q1", "q1", "q2", "q2"],
                "model_id": ["m"] * 4,
                "score": [1.0, 0.0, 1.0, 1.0],
                "epoch": [0, 1, 0, 1],
                "cluster_id": ["phys", "phys", "bio", "bio"],
                "correct": [True, False, True, True],
            }
        )
        results = from_dataframe(df)

        assert len(results) == 4
        assert results[0].item_id == "q1"
        assert results[1].epoch == 1
        assert results[2].cluster_id == "bio"
        assert results[1].correct is False

    def test_column_remap(self):
        df = pd.DataFrame(
            {"qid": ["q1", "q2"], "system": ["m", "m"], "acc": [1.0, 0.0]}
        )
        results = from_dataframe(df, item_id="qid", model_id="system", score="acc")

        assert results[0].score == 1.0
        assert results[0].epoch == 0
        assert results[0].cluster_id is None

    def test_score_derived_from_correct(self):
        df = pd.DataFrame(
            {"item_id": ["q1", "q2"], "model_id": ["m", "m"], "correct": [True, False]}
        )
        results = from_dataframe(df)

        assert [r.score for r in results] == [1.0, 0.0]

    def test_missing_required_column_raises(self):
        df = pd.DataFrame({"item_id": ["q1"], "score": [1.0]})

        with pytest.raises(ValueError, match="model_id"):
            from_dataframe(df)

    def test_no_score_source_raises(self):
        df = pd.DataFrame({"item_id": ["q1"], "model_id": ["m"]})

        with pytest.raises(ValueError, match="score"):
            from_dataframe(df)


def fake_log(samples, model="openai/gpt-test"):
    return SimpleNamespace(eval=SimpleNamespace(model=model), samples=samples)


def fake_sample(sid, value, *, epoch=1, scorer="choice", metadata=None):
    return SimpleNamespace(
        id=sid,
        epoch=epoch,
        scores={scorer: SimpleNamespace(value=value)},
        metadata=metadata or {},
    )


class TestFromInspect:
    def test_categorical_values(self):
        log = fake_log([fake_sample("q1", "C"), fake_sample("q2", "I")])
        results = from_inspect(log)

        assert results[0].score == 1.0 and results[0].correct is True
        assert results[1].score == 0.0 and results[1].correct is False
        assert results[0].model_id == "openai/gpt-test"

    def test_partial_and_numeric_values(self):
        log = fake_log([fake_sample("q1", "P"), fake_sample("q2", 0.3)])
        results = from_inspect(log)

        assert results[0].score == 0.5 and results[0].correct is None
        assert results[1].score == 0.3 and results[1].correct is None

    def test_epochs_preserved(self):
        log = fake_log([fake_sample("q1", "C", epoch=1), fake_sample("q1", "I", epoch=2)])
        results = from_inspect(log)

        assert [r.epoch for r in results] == [1, 2]

    def test_cluster_field_from_metadata(self):
        log = fake_log([fake_sample("q1", "C", metadata={"domain": "physics"})])
        results = from_inspect(log, cluster_field="domain")

        assert results[0].cluster_id == "physics"

    def test_multiple_scorers_require_choice(self):
        sample = SimpleNamespace(
            id="q1",
            epoch=1,
            scores={
                "choice": SimpleNamespace(value="C"),
                "model_graded": SimpleNamespace(value="I"),
            },
            metadata={},
        )
        log = fake_log([sample])

        with pytest.raises(ValueError, match="scorer="):
            from_inspect(log)

        results = from_inspect(log, scorer="model_graded")
        assert results[0].score == 0.0

    def test_empty_log_raises(self):
        with pytest.raises(ValueError, match="no samples"):
            from_inspect(fake_log([]))

    def test_unknown_categorical_raises(self):
        log = fake_log([fake_sample("q1", "X")])

        with pytest.raises(ValueError, match="Unrecognized"):
            from_inspect(log)


class TestFromInspectRealLog:
    """Integration against a genuine Inspect .eval log when one exists locally.

    Generation logs live in logs/ (gitignored — they embed the gated GPQA
    questions verbatim), so this class is skipped on CI and on fresh clones.
    """

    @pytest.fixture()
    def real_log(self):
        from pathlib import Path

        logs = sorted((Path(__file__).parent.parent / "logs").rglob("*.eval"))
        if not logs:
            pytest.skip("no local .eval logs (generation has not been run here)")
        return logs[0]

    def test_parses_real_eval_log(self, real_log):
        results = from_inspect(str(real_log))

        assert len(results) >= 2
        items = {r.item_id for r in results}
        epochs = {r.epoch for r in results}
        # rectangular: every item scored once per epoch
        assert len(results) == len(items) * len(epochs)
        assert all(r.score in (0.0, 1.0) for r in results)
        assert all(r.correct in (True, False) for r in results)
        assert len({r.model_id for r in results}) == 1
