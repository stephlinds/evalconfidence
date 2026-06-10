"""Tests for power(): closed-form checks and empirical calibration vs compare()."""

import math

import numpy as np
import pytest
from scipy import stats

from evalconfidence import ItemResult, compare, power

Z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)  # ~2.8016 at alpha=.05, power=.8


def results_from(scores, model, item_prefix="q", cluster_ids=None):
    return [
        ItemResult(
            item_id=f"{item_prefix}{i}",
            model_id=model,
            score=float(s),
            cluster_id=None if cluster_ids is None else cluster_ids[i],
        )
        for i, s in enumerate(scores)
    ]


class TestClosedForm:
    def test_required_n_binary_50pct_unpaired(self):
        # var = 2*p(1-p) = 0.5 at p=.5; n = Z^2 * 0.5 / 0.1^2
        res = power(baseline_var=0.5, mde=0.10)

        assert res.design == "specified"
        assert res.n_required == math.ceil(Z**2 * 0.5 / 0.01)  # 393

    def test_mde_at_gpqa_size_matches_docs(self):
        # docs/why-it-works.md: N=198 unpaired at 50% accuracy -> MDE ~14 points
        res = power(baseline_var=0.5, n=198)

        assert res.mde_detectable == pytest.approx(Z * math.sqrt(0.5 / 198), rel=1e-9)
        assert 0.13 < res.mde_detectable < 0.15

    def test_round_trip_n_and_mde(self):
        n_req = power(baseline_var=0.3, mde=0.05).n_required
        mde_back = power(baseline_var=0.3, n=n_req).mde_detectable

        assert mde_back <= 0.05  # ceil(n) can only make the eval more sensitive
        assert mde_back == pytest.approx(0.05, rel=0.02)

    def test_achieved_power_at_planned_mde(self):
        # At exactly the planned MDE and n, achieved power ~= target
        res = power(baseline_var=0.5, n=198, mde=Z * math.sqrt(0.5 / 198))

        assert res.power_achieved == pytest.approx(0.8, abs=1e-9)


class TestPilotVariance:
    def test_paired_pilot_uses_difference_variance(self):
        rng = np.random.default_rng(0)
        d = rng.normal(0.02, 0.1, 120)
        b = np.full(120, 0.5)
        ra = results_from(b + d, "A")
        rb = results_from(b, "B")

        res = power((ra, rb), mde=0.03)

        expected_var = float(np.var(d, ddof=1))
        assert res.design == "paired"
        assert res.variance == pytest.approx(expected_var, rel=1e-9)
        assert res.n_required == math.ceil(Z**2 * expected_var / 0.03**2)

    def test_single_pilot_doubles_item_variance(self):
        rng = np.random.default_rng(1)
        scores = rng.uniform(0, 1, 150)
        res = power(results_from(scores, "A"), n=200)

        expected_var = 2 * float(np.var(scores, ddof=1))
        assert res.design == "unpaired"
        assert res.variance == pytest.approx(expected_var, rel=1e-9)
        assert any("paired design would need fewer" in note for note in res.notes)

    def test_cluster_inflation_increases_required_n(self):
        rng = np.random.default_rng(2)
        n_clusters, per_cluster = 30, 5
        cluster_effect = np.repeat(rng.normal(0, 0.15, n_clusters), per_cluster)
        scores = 0.6 + cluster_effect + rng.normal(0, 0.08, n_clusters * per_cluster)
        cluster_ids = [f"c{i // per_cluster}" for i in range(n_clusters * per_cluster)]

        clustered = power(results_from(scores, "A", cluster_ids=cluster_ids), mde=0.05)
        flat = power(results_from(scores, "A"), mde=0.05)

        assert clustered.design_effect is not None and clustered.design_effect > 1.3
        assert clustered.n_required > flat.n_required
        assert any("design-effect" in line for line in clustered.summary().splitlines())

    def test_epochs_aggregated_note(self):
        rng = np.random.default_rng(3)
        results = [
            ItemResult(f"q{i}", "A", float(rng.random() < 0.7), epoch=k)
            for i in range(60)
            for k in range(3)
        ]
        res = power(results, n=100)

        assert any("aggregated" in note for note in res.notes)


class TestEmpiricalCalibration:
    def test_planned_n_delivers_target_power_via_compare(self):
        """The loop closed: run compare() at the planned n; rejection rate ~= 80%."""
        rng = np.random.default_rng(4)
        sigma_d, true_gap = 0.1, 0.03
        n_req = power(baseline_var=sigma_d**2, mde=true_gap).n_required  # 88

        rejections = 0
        reps = 400
        for _ in range(reps):
            d = rng.normal(true_gap, sigma_d, n_req)
            ra = results_from(0.5 + d, "A")
            rb = results_from(np.full(n_req, 0.5), "B")
            rejections += compare(ra, rb).significant

        assert 0.72 <= rejections / reps <= 0.88

    def test_underpowered_eval_misses_the_gap(self):
        """At a quarter of the required n, power collapses well below target."""
        rng = np.random.default_rng(5)
        sigma_d, true_gap = 0.1, 0.03
        n_req = power(baseline_var=sigma_d**2, mde=true_gap).n_required
        n_small = n_req // 4

        rejections = 0
        reps = 400
        for _ in range(reps):
            d = rng.normal(true_gap, sigma_d, n_small)
            ra = results_from(0.5 + d, "A")
            rb = results_from(np.full(n_small, 0.5), "B")
            rejections += compare(ra, rb).significant

        assert rejections / reps < 0.5


class TestValidation:
    def test_requires_exactly_one_variance_source(self):
        with pytest.raises(ValueError, match="exactly one variance source"):
            power(mde=0.05)
        with pytest.raises(ValueError, match="exactly one variance source"):
            power(results_from([0.1, 0.9], "A"), baseline_var=0.3, mde=0.05)

    def test_requires_mde_or_n(self):
        with pytest.raises(ValueError, match="mde"):
            power(baseline_var=0.5)

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError, match="positive"):
            power(baseline_var=0.5, mde=-0.1)
        with pytest.raises(ValueError, match="at least 2"):
            power(baseline_var=0.5, n=1)
        with pytest.raises(ValueError, match="positive"):
            power(baseline_var=0.0, mde=0.05)

    def test_zero_pilot_variance_raises(self):
        ra = results_from([0.5, 0.5, 0.5], "A")

        with pytest.raises(ValueError, match="variance is 0"):
            power(ra, mde=0.05)

    def test_summary_for_each_direction(self):
        n_text = power(baseline_var=0.5, mde=0.10).summary()
        mde_text = power(baseline_var=0.5, n=198).summary()
        pow_text = power(baseline_var=0.5, n=198, mde=0.05).summary()

        assert "requires ~" in n_text
        assert "minimum detectable gap" in mde_text
        assert "% power" in pow_text
