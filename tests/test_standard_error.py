"""Synthetic tests with a known data-generating process.

The DGP is a one-way random-effects model: score_ik = mu + a_i + e_ik with
item effect a_i ~ N(0, sigma_a^2) and noise e_ik ~ N(0, sigma_e^2), K epochs
per item. Theory gives ICC rho = sigma_a^2 / (sigma_a^2 + sigma_e^2) and a
design effect of 1 + (K - 1) * rho for clustering by item.
"""

import numpy as np
import pytest

from evalconfidence import ItemResult, standard_error


def make_results(scores_by_item: np.ndarray, *, model="m", cluster_ids=None):
    """scores_by_item: (n_items, n_epochs) array -> flat list of ItemResults."""
    n_items, n_epochs = scores_by_item.shape
    results = []
    for i in range(n_items):
        for k in range(n_epochs):
            results.append(
                ItemResult(
                    item_id=f"q{i}",
                    model_id=model,
                    score=float(scores_by_item[i, k]),
                    epoch=k,
                    cluster_id=None if cluster_ids is None else cluster_ids[i],
                )
            )
    return results


def random_effects_dgp(rng, n_items, n_epochs, sigma_a, sigma_e, mu=0.5):
    a = rng.normal(0, sigma_a, size=(n_items, 1))
    e = rng.normal(0, sigma_e, size=(n_items, n_epochs))
    return mu + a + e


class TestTemperatureZeroExtreme:
    """Identical epochs carry zero new information: DEFF = (nK-1)/(n-1) ~= K."""

    def test_design_effect_exact(self):
        rng = np.random.default_rng(0)
        n_items, n_epochs = 50, 5
        item_scores = rng.uniform(0, 1, size=(n_items, 1))
        scores = np.repeat(item_scores, n_epochs, axis=1)

        res = standard_error(make_results(scores))

        expected_deff = (n_items * n_epochs - 1) / (n_items - 1)
        assert res.cluster_by == "item"
        assert res.n_clusters == n_items
        assert res.design_effect == pytest.approx(expected_deff, rel=1e-9)

    def test_clustered_se_equals_item_level_se(self):
        rng = np.random.default_rng(1)
        n_items = 200
        item_scores = rng.uniform(0, 1, size=(n_items, 1))
        scores = np.repeat(item_scores, 5, axis=1)

        res = standard_error(make_results(scores))
        item_level = standard_error(make_results(item_scores))

        assert res.se_clustered == pytest.approx(item_level.se_naive, rel=1e-9)


class TestDesignEffectTheory:
    def test_inflation_matches_icc(self):
        rng = np.random.default_rng(2)
        n_items, n_epochs = 5000, 5
        sigma_a, sigma_e = 0.3, 0.3  # rho = 0.5
        rho = sigma_a**2 / (sigma_a**2 + sigma_e**2)
        scores = random_effects_dgp(rng, n_items, n_epochs, sigma_a, sigma_e)

        res = standard_error(make_results(scores))

        expected_deff = 1 + (n_epochs - 1) * rho
        assert res.design_effect == pytest.approx(expected_deff, rel=0.10)

    def test_independent_observations_no_inflation(self):
        rng = np.random.default_rng(3)
        scores = random_effects_dgp(rng, 5000, 5, sigma_a=0.0, sigma_e=0.4)

        res = standard_error(make_results(scores))

        assert res.design_effect == pytest.approx(1.0, abs=0.1)


class TestCICoverage:
    """Clustered CI covers the true mean ~95% of the time; naive badly under-covers."""

    def test_coverage(self):
        rng = np.random.default_rng(4)
        reps, n_items, n_epochs = 500, 150, 5
        sigma_a, sigma_e, mu = 0.4, 0.25, 0.5  # rho ~ 0.72, DEFF ~ 3.9

        cover_naive = cover_clustered = 0
        for _ in range(reps):
            scores = random_effects_dgp(rng, n_items, n_epochs, sigma_a, sigma_e, mu)
            res = standard_error(make_results(scores))
            cover_naive += res.ci_naive[0] <= mu <= res.ci_naive[1]
            cover_clustered += res.ci_clustered[0] <= mu <= res.ci_clustered[1]

        assert 0.92 <= cover_clustered / reps <= 0.98
        assert cover_naive / reps < 0.85


class TestClusterResolution:
    def test_single_epoch_falls_back_to_cluster_id(self):
        rng = np.random.default_rng(5)
        scores = rng.uniform(0, 1, size=(60, 1))
        cluster_ids = [f"domain{i % 3}" for i in range(60)]

        res = standard_error(make_results(scores, cluster_ids=cluster_ids))

        assert res.cluster_by == "cluster_id"
        assert res.n_clusters == 3

    def test_epochs_take_priority_over_cluster_id(self):
        rng = np.random.default_rng(6)
        scores = rng.uniform(0, 1, size=(30, 3))
        cluster_ids = [f"domain{i % 3}" for i in range(30)]

        res = standard_error(make_results(scores, cluster_ids=cluster_ids))

        assert res.cluster_by == "item"

    def test_no_structure_returns_naive_only(self):
        rng = np.random.default_rng(7)
        res = standard_error(make_results(rng.uniform(0, 1, size=(50, 1))))

        assert res.se_clustered is None
        assert res.se == res.se_naive
        assert res.ci == res.ci_naive

    def test_cluster_none_disables_clustering(self):
        rng = np.random.default_rng(8)
        scores = np.repeat(rng.uniform(0, 1, size=(50, 1)), 5, axis=1)

        res = standard_error(make_results(scores), cluster=None)

        assert res.se_clustered is None

    def test_missing_cluster_id_raises(self):
        rng = np.random.default_rng(9)
        results = make_results(rng.uniform(0, 1, size=(10, 1)))

        with pytest.raises(ValueError, match="cluster_id=None"):
            standard_error(results, cluster="cluster_id")

    def test_unknown_cluster_option_raises(self):
        rng = np.random.default_rng(10)
        results = make_results(rng.uniform(0, 1, size=(10, 1)))

        with pytest.raises(ValueError, match="Unknown cluster option"):
            standard_error(results, cluster="epoch")


class TestGuardrails:
    def test_two_models_raise(self):
        results = [
            ItemResult(item_id="q1", model_id="a", score=1.0),
            ItemResult(item_id="q1", model_id="b", score=0.0),
        ]
        with pytest.raises(ValueError, match="single system"):
            standard_error(results)

    def test_too_few_results_raise(self):
        with pytest.raises(ValueError, match="at least 2"):
            standard_error([ItemResult(item_id="q1", model_id="a", score=1.0)])

    def test_summary_mentions_both_ses(self):
        rng = np.random.default_rng(11)
        scores = np.repeat(rng.uniform(0, 1, size=(20, 1)), 3, axis=1)

        text = standard_error(make_results(scores)).summary()

        assert "Naive i.i.d. SE" in text
        assert "Cluster-robust SE" in text
        assert "Inflation" in text
