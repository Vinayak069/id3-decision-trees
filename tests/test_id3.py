"""Unit tests: each one checks a specific claim or number from Quinlan (1986).

    python -m pytest -q tests
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import id3  # noqa: E402
from simulate import TRUE_TREE_TEXT, simulate  # noqa: E402


@pytest.fixture(scope="module")
def weather():
    d = pd.read_csv(os.path.join(ROOT, "data/real/weather_quinlan1986.csv"), dtype=str)
    return d[["outlook", "temperature", "humidity", "windy"]], d["class"]


def test_section4_information_of_table1():
    assert id3.entropy([9, 5]) == pytest.approx(0.940, abs=5e-4)          # paper: 0.940 bits


def test_section4_gains_of_table1(weather):
    X, y = weather
    paper = {"outlook": 0.246, "temperature": 0.029, "humidity": 0.151, "windy": 0.048}
    for a, g in paper.items():
        # the paper rounds intermediate results (0.940 - 0.694), so allow 1e-3
        assert id3.information_gain(pd.crosstab(X[a], y).to_numpy()) == pytest.approx(g, abs=1.1e-3)


def test_section4_tree_is_figure2(weather):
    X, y = weather
    t = id3.ID3().fit(X, y)
    assert t.root_.attribute == "outlook"
    assert t.root_.children["sunny"].attribute == "humidity"
    assert t.root_.children["rain"].attribute == "windy"
    assert t.root_.children["overcast"].is_leaf
    assert t.score(X, y) == 1.0 and t.n_leaves == 5


def test_section7_gain_ratio_numbers(weather):
    X, y = weather
    t = pd.crosstab(X["outlook"], y).to_numpy()
    assert id3.intrinsic_value(t) == pytest.approx(1.578, abs=1e-3)
    assert id3.gain_ratio(t) == pytest.approx(0.156, abs=1e-3)
    h = pd.crosstab(X["humidity"], y).to_numpy()
    assert id3.gain_ratio(h) == pytest.approx(0.151, abs=1.1e-3)


def test_footnote3_gain_is_mutual_information():
    rng = np.random.default_rng(0)
    t = rng.integers(0, 20, size=(4, 3)).astype(float)
    p = t / t.sum()
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = np.nansum(p * np.log2(p / (px * py)))
    assert id3.information_gain(t) == pytest.approx(mi, abs=1e-12)


def test_section5_chi_square_matches_scipy():
    rng = np.random.default_rng(1)
    for _ in range(20):
        t = rng.integers(1, 30, size=(rng.integers(2, 5), rng.integers(2, 4))).astype(float)
        stat, p = id3.chi_square(t)
        s2, p2, _, _ = chi2_contingency(t, correction=False)
        assert stat == pytest.approx(s2) and p == pytest.approx(p2)


def test_section7_splitting_a_value_never_lowers_gain():
    """Paper: gain(A') >= gain(A) when A' splits one value of A into two."""
    rng = np.random.default_rng(2)
    for _ in range(500):
        t = rng.integers(0, 15, size=(3, 2)).astype(float)
        cut = rng.integers(0, t[0] + 1)
        t2 = np.vstack([cut, t[0] - cut, t[1:]])
        assert id3.information_gain(t2) >= id3.information_gain(t) - 1e-12


def test_section6_unknowns_can_only_lower_gain():
    """Paper: the proportional treatment of unknown values can only decrease the gain."""
    rng = np.random.default_rng(3)
    for _ in range(500):
        t = rng.integers(1, 20, size=(3, 2)).astype(float)
        u = rng.integers(0, 10, size=2).astype(float)
        assert id3.information_gain(id3._adjust(t, u)) <= id3.information_gain(t) + 1e-12


def test_section6_unknown_as_value_anomaly():
    """Paper's example: E(A) = 1 bit, but E(A') = 0.84 bits once one value becomes 'unknown'."""
    assert id3.expected_information([[2, 2], [2, 2]]) == pytest.approx(1.0)
    assert id3.expected_information([[1, 2], [2, 2], [1, 0]]) == pytest.approx(0.84, abs=0.01)


def test_section6_token_splitting_weights(weather):
    X, y = weather
    t = id3.ID3().fit(X, y)
    row = pd.DataFrame([{"outlook": "?", "temperature": "hot", "humidity": "high", "windy": "false"}])
    # sunny (5/14) -> N, overcast (4/14) -> P, rain (5/14) -> P (windy = false)
    p = t.predict_proba(row)[0]
    assert p[list(t.classes_).index("P")] == pytest.approx(9 / 14)


def test_simulated_tree_is_recovered():
    df = simulate(2000, seed=2026)
    t = id3.ID3().fit(df.drop(columns="class"), df["class"])
    text = "\n".join(l.split("  (n=")[0] for l in t.export_text().split("\n"))
    assert text.strip() == TRUE_TREE_TEXT.strip()


def test_chi2_test_blocks_irrelevant_attributes():
    rng = np.random.default_rng(4)
    X = pd.DataFrame({f"R{i}": rng.choice(list("abc"), 500) for i in range(5)})
    y = rng.choice(["P", "N"], 500)
    assert id3.ID3(chi2_alpha=0.01).fit(X, y).n_nodes == 1     # pure noise -> single leaf
    assert id3.ID3().fit(X, y).n_nodes > 50                    # without the test it fits the noise


def test_threshold_at_midpoint():
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0]})
    y = ["N", "N", "N", "P", "P", "P"]
    t = id3.ID3(continuous=["x"]).fit(X, y)
    assert t.root_.kind == "threshold" and t.root_.param == pytest.approx(6.5)
    assert list(t.predict(pd.DataFrame({"x": [0.0, 6.0, 7.0, 99.0]}))) == ["N", "N", "P", "P"]


def test_subset_split_is_binary_and_exact():
    X = pd.DataFrame({"c": list("abcdab")})
    y = ["P", "N", "P", "N", "P", "N"]          # a, c -> P ; b, d -> N
    t = id3.ID3(split="subset").fit(X, y)
    assert t.root_.kind == "subset" and set(t.root_.children) == {"in", "out"}
    assert t.score(X, y) == 1.0 and t.n_nodes == 3


def test_fit_does_not_modify_input(weather):
    X, y = weather
    Xu = X.copy(); Xu.iloc[0, 0] = "?"
    before = Xu.copy()
    id3.ID3().fit(Xu, y)
    pd.testing.assert_frame_equal(Xu, before)
