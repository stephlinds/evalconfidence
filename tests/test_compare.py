"""Tests for compare(): correctness against references and a known DGP."""

import numpy as np
import pytest
from scipy import stats

from evalconfidence import ItemResult, compare


def results_from(scores, model, item_prefix="q", epoch=0):
    return [
        ItemResult(item_id=f"{item_prefix}{i}", model_id=model, score=float(s), epoch=epoch)
        for i, s in enumerate(scores)
    ]


def correlated_pair_dgp(rng, n_items, true_gap, sigma_item=0.25, sigma_noise=0.08):
    """Continuous scores with shared item difficulty: high cross-model Cov."""
    difficulty = rng.normal(0.6, sigma_item, n_items)
    a = difficulty + true_gap + rng.normal(0, sigma_noise, n_items)
    b = difficulty + rng.normal(0, sigma_noise, n_items)
    return a, b


class TestPairedContinuous:
    def test_wrong_winner_scenario(self):
        """Figure 1: paired detects a real gap the unpaired analysis buries."""
        rng = np.random.default_rng(0)
        a, b = correlated_pair_dgp(
            rng, n_items=200, true_gap=0.035, sigma_item=0.35, sigma_noise=0.05
        )

        paired = compare(results_from(a, "A"), results_from(b, "B"))
        unpaired = compare(results_from(a, "A"), results_from(b, "B"), paired=False)

        assert paired.paired and paired.method == "paired_t"
        assert paired.significant
        assert not unpaired.significant
        assert paired.variance_reduction > 5  # shared difficulty dominates noise
        assert paired.effective_n > paired.n_pairs

    def test_matches_scipy_ttest_rel(self):
        rng = np.random.default_rng(1)
        a, b = correlated_pair_dgp(rng, n_items=80, true_gap=0.02)

        res = compare(results_from(a, "A"), results_from(b, "B"))
        ref = stats.ttest_rel(a, b)

        assert res.p_value == pytest.approx(ref.pvalue, rel=1e-9)
        assert res.diff == pytest.approx(float(np.mean(a - b)), rel=1e-9)
        lo, hi = ref.confidence_interval(0.95)
        assert res.ci_diff == pytest.approx((lo, hi), rel=1e-9)

    def test_ci_coverage_under_known_dgp(self):
        rng = np.random.default_rng(2)
        true_gap, reps = 0.02, 400
        covered = 0
        for _ in range(reps):
            a, b = correlated_pair_dgp(rng, n_items=60, true_gap=true_gap)
            res = compare(results_from(a, "A"), results_from(b, "B"))
            covered += res.ci_diff[0] <= true_gap <= res.ci_diff[1]
        assert 0.92 <= covered / reps <= 0.98

    def test_epochs_aggregated_to_item_means(self):
        rng = np.random.default_rng(3)
        results_a, results_b = [], []
        for i in range(50):
            for k in range(3):
                results_a.append(ItemResult(f"q{i}", "A", float(rng.random() < 0.8), epoch=k))
                results_b.append(ItemResult(f"q{i}", "B", float(rng.random() < 0.7), epoch=k))

        res = compare(results_a, results_b)

        assert res.method == "paired_t"  # binary but multi-epoch -> continuous path
        assert any("aggregated" in n for n in res.notes)
        assert res.n_pairs == 50


class TestMcNemar:
    def make_binary(self, n00, n01, n10, n11):
        xa = [1] * n10 + [0] * n01 + [1] * n11 + [0] * n00
        xb = [0] * n10 + [1] * n01 + [1] * n11 + [0] * n00
        return results_from(xa, "A"), results_from(xb, "B")

    def test_exact_binomial_hand_computed(self):
        # n10=10, n01=2: two-sided exact p = 2*P(X<=2 | n=12, p=.5) = 158/4096
        ra, rb = self.make_binary(n00=40, n01=2, n10=10, n11=148)
        res = compare(ra, rb)

        assert res.method == "mcnemar"
        assert res.n_discordant == 12
        assert res.p_value == pytest.approx(2 * (1 + 12 + 66) / 4096, rel=1e-9)
        assert res.odds_ratio == pytest.approx(5.0)
        assert res.diff == pytest.approx(8 / 200)

    def test_chi_square_for_many_discordant(self):
        ra, rb = self.make_binary(n00=50, n01=30, n10=60, n11=60)
        res = compare(ra, rb)

        # continuity-corrected chi2 = (|60-30|-1)^2 / 90
        expected_p = stats.chi2.sf((30 - 1) ** 2 / 90, df=1)
        assert res.p_value == pytest.approx(expected_p, rel=1e-9)

    def test_zero_cell_odds_ratio_finite(self):
        ra, rb = self.make_binary(n00=50, n01=0, n10=8, n11=142)
        res = compare(ra, rb)

        assert np.isfinite(res.odds_ratio)
        assert res.significant

    def test_perfect_agreement(self):
        ra, rb = self.make_binary(n00=100, n01=0, n10=0, n11=100)
        res = compare(ra, rb)

        assert res.p_value == 1.0
        assert res.diff == 0.0


class TestUnpairedFallback:
    def test_welch_matches_scipy(self):
        rng = np.random.default_rng(4)
        a = rng.normal(0.7, 0.1, 90)
        b = rng.normal(0.65, 0.15, 110)

        res = compare(
            results_from(a, "A", item_prefix="qa"),
            results_from(b, "B", item_prefix="qb"),
        )
        ref = stats.ttest_ind(a, b, equal_var=False)

        assert res.method == "welch_t"
        assert not res.paired
        assert res.p_value == pytest.approx(ref.pvalue, rel=1e-9)
        assert any("Pairing unavailable" in w for w in res.warnings)

    def test_two_proportion_for_disjoint_binary(self):
        rng = np.random.default_rng(5)
        a = (rng.random(200) < 0.8).astype(float)
        b = (rng.random(200) < 0.6).astype(float)

        res = compare(
            results_from(a, "A", item_prefix="qa"),
            results_from(b, "B", item_prefix="qb"),
        )

        assert res.method == "two_proportion"
        assert res.significant
        assert res.ci_diff[0] < res.diff < res.ci_diff[1]

    def test_forced_unpaired_warns_about_waste(self):
        rng = np.random.default_rng(6)
        a, b = correlated_pair_dgp(rng, n_items=50, true_gap=0.02)

        res = compare(results_from(a, "A"), results_from(b, "B"), paired=False)

        assert not res.paired
        assert any("paired=False" in w for w in res.warnings)


class TestAutoPairing:
    def test_partial_overlap_below_threshold_falls_back(self):
        rng = np.random.default_rng(7)
        a = rng.random(100)
        b = rng.random(100)
        ra = results_from(a, "A")  # q0..q99
        rb = [
            ItemResult(item_id=f"q{i + 50}", model_id="B", score=float(s))
            for i, s in enumerate(b)
        ]  # q50..q149: 50% overlap

        res = compare(ra, rb)

        assert not res.paired
        assert any("50%" in w for w in res.warnings)

    def test_overlap_above_threshold_pairs_and_notes_drops(self):
        rng = np.random.default_rng(8)
        a, b = correlated_pair_dgp(rng, n_items=100, true_gap=0.02)
        ra = results_from(a, "A")
        rb = results_from(b, "B")[:95]  # 95% coverage

        res = compare(ra, rb)

        assert res.paired
        assert res.n_pairs == 95
        assert any("dropped" in n for n in res.notes)

    def test_paired_true_without_shared_items_raises(self):
        ra = results_from([1.0, 0.0], "A", item_prefix="qa")
        rb = results_from([1.0, 0.0], "B", item_prefix="qb")

        with pytest.raises(ValueError, match="shared item_id"):
            compare(ra, rb, paired=True)

    def test_invalid_paired_value_raises(self):
        ra = results_from([1.0, 0.0], "A")
        rb = results_from([1.0, 0.0], "B")

        with pytest.raises(ValueError, match="paired must be"):
            compare(ra, rb, paired="yes")


class TestGuardrailsAndOutput:
    def test_mixed_models_in_one_argument_raise(self):
        ra = results_from([1.0, 0.0], "A") + results_from([1.0], "C", item_prefix="z")
        rb = results_from([1.0, 0.0], "B")

        with pytest.raises(ValueError, match="mixes models"):
            compare(ra, rb)

    def test_same_label_both_sides_noted(self):
        rng = np.random.default_rng(9)
        a, b = correlated_pair_dgp(rng, n_items=30, true_gap=0.0)

        res = compare(results_from(a, "M"), results_from(b, "M"))

        assert any("share a label" in n for n in res.notes)

    def test_summary_leads_with_decision(self):
        rng = np.random.default_rng(10)
        a, b = correlated_pair_dgp(rng, n_items=150, true_gap=0.03)
        a, b = np.clip(a, 0, 1), np.clip(b, 0, 1)  # unit scale -> "points" output

        text = compare(results_from(a, "A"), results_from(b, "B")).summary()

        assert "outperform" in text
        assert "points" in text
        assert "Pairing reduced the comparison variance" in text

    def test_identical_differences_degenerate_se(self):
        ra = results_from([0.5, 0.6, 0.7], "A")
        rb = results_from([0.4, 0.5, 0.6], "B")

        res = compare(ra, rb)

        assert res.diff == pytest.approx(0.1)
        assert res.p_value == 0.0
        assert any("identical" in n for n in res.notes)
