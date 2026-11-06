"""
id3.py -- a from-scratch implementation of Quinlan's ID3 (Machine Learning 1:81-106, 1986).

Everything the paper describes is implemented with NumPy/pandas only:

  Section 4  entropy I(p, n), expected information E(A), gain(A); divide-and-conquer tree;
             empty branch -> 'null' leaf (original ID3) or majority class of the parent;
             the windowing loop.
  Section 5  noise: majority-class leaves for inadequate attributes, and the chi-square
             test of independence that refuses to split on an attribute whose irrelevance
             cannot be rejected (e.g. at 99 % confidence).
  Section 6  unknown attribute values: proportional redistribution when computing gain,
             unknowns discarded below the chosen split, and 'token' splitting at prediction.
  Section 7  gain ratio = gain / IV(A), chosen among attributes with at least average gain.

  Section 7  also the binary value-subset tests (ASSISTANT) and thresholds for continuous
             attributes at midpoints of consecutive values (split='subset', continuous=[...]).

Extra knobs that are not in the paper (labelled as such): max_depth and min_samples_split.
These are the "depth control" studied in the project.

The paper uses two classes (P, N); here every formula is written for K >= 2 classes,
which reduces to the paper's formulas when K = 2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2 as _chi2

UNKNOWN_TOKENS = {"?", "", "nan", "NaN", "None"}


# ----------------------------------------------------------------------------------------
# Information measures (Section 4 and 7)
# ----------------------------------------------------------------------------------------
def entropy(counts) -> float:
    """I(p, n) generalised to K classes: -sum_k (c_k/N) log2 (c_k/N), with 0 log 0 = 0."""
    c = np.asarray(counts, dtype=float)
    total = c.sum()
    if total <= 0:
        return 0.0
    q = c[c > 0] / total
    return float(-(q * np.log2(q)).sum())


def expected_information(table) -> float:
    """E(A) = sum_i (|C_i|/|C|) I(C_i) for a values x classes count table."""
    t = np.asarray(table, dtype=float)
    total = t.sum()
    if total <= 0:
        return 0.0
    return float(sum(row.sum() / total * entropy(row) for row in t))


def information_gain(table) -> float:
    """gain(A) = I(C) - E(A)."""
    t = np.asarray(table, dtype=float)
    return entropy(t.sum(axis=0)) - expected_information(t)


def intrinsic_value(table) -> float:
    """IV(A) = -sum_i (|C_i|/|C|) log2 (|C_i|/|C|), the information in the attribute's value."""
    return entropy(np.asarray(table, dtype=float).sum(axis=1))


def gain_ratio(table) -> float:
    iv = intrinsic_value(table)
    return information_gain(table) / iv if iv > 0 else 0.0


def chi_square(table):
    """Quinlan's Section-5 statistic sum_i sum_k (c_ik - c'_ik)^2 / c'_ik and its p-value.

    c'_ik = c_k * |C_i| / |C| is the count expected if A were irrelevant to the class.
    Degrees of freedom (v-1)(K-1); the paper's two-class case gives v-1.
    """
    t = np.asarray(table, dtype=float)
    t = t[t.sum(axis=1) > 0]            # values that actually occur
    t = t[:, t.sum(axis=0) > 0]         # classes that actually occur
    if t.shape[0] < 2 or t.shape[1] < 2:
        return 0.0, 1.0
    expected = np.outer(t.sum(axis=1), t.sum(axis=0)) / t.sum()
    stat = float(((t - expected) ** 2 / expected).sum())
    dof = (t.shape[0] - 1) * (t.shape[1] - 1)
    return stat, float(_chi2.sf(stat, dof))


# ----------------------------------------------------------------------------------------
# Tree structure
# ----------------------------------------------------------------------------------------
@dataclass
class Node:
    dist: np.ndarray                      # class counts that reached this node in training
    attribute: Optional[str] = None       # None -> leaf
    kind: str = "value"                   # 'value' (multiway), 'subset' (binary, Section 7) or 'threshold'
    param: object = None                  # frozenset of values for 'subset', cut point for 'threshold'
    children: Dict[str, "Node"] = field(default_factory=dict)
    ratios: Dict[str, float] = field(default_factory=dict)  # |C_i|/sum|C_j| for token splitting
    is_null: bool = False                 # an empty-branch leaf in 'null' mode
    gain: float = 0.0
    p_value: float = 1.0

    @property
    def is_leaf(self) -> bool:
        return self.attribute is None

    def branch_of(self, v):
        """Branch key for a known value v, or None if the value was never seen here."""
        if self.kind == "value":
            return v if v in self.children else None
        if self.kind == "subset":
            return "in" if v in self.param else ("out" if v in self._out else None)
        return "le" if float(v) <= self.param else "gt"

    def test_text(self, key) -> str:
        if self.kind == "value":
            return f"{self.attribute} = {key}"
        if self.kind == "subset":
            vals = ",".join(sorted(self.param)) if key == "in" else ",".join(sorted(self._out))
            return f"{self.attribute} in {{{vals}}}"
        return f"{self.attribute} {'<=' if key == 'le' else '>'} {self.param:.4g}"


def _is_unknown(series: pd.Series) -> np.ndarray:
    s = series.astype(object)
    return (s.isna() | s.astype(str).isin(UNKNOWN_TOKENS)).to_numpy()


def _adjust(table, unknown_counts):
    """Section 6: spread the unknown objects over the outcomes in proportion to |C_i|."""
    if unknown_counts.sum() == 0 or table.sum() == 0:
        return table
    ratio = table.sum(axis=1) / table.sum()
    return table + np.outer(ratio, unknown_counts)


class ID3:
    """ID3 decision tree.

    Parameters
    ----------
    criterion : 'gain' (Section 4) or 'gain_ratio' (Section 7).
    chi2_alpha : None disables the Section-5 test. Otherwise an attribute is only tested when
        the chi-square p-value is below alpha (alpha = 0.01 is the paper's 99 % confidence).
    split : 'multiway' -- one branch per value (ID3, Section 4);
            'subset'   -- the binary value-subset tests of ASSISTANT discussed in Section 7.
              All 2^(v-1)-1 subsets are searched when v <= max_subset_values; above that, values are
              ordered by their class-1 proportion and only the v-1 cuts of that order are tried,
              which is exact for two classes (Breiman et al. 1984) and a heuristic otherwise.
    continuous : names of numeric attributes, split by a threshold at the midpoint of two
        consecutive observed values (Section 7). Binary tests may reuse an attribute below them.
    max_depth, min_samples_split : depth control (not in the paper; added for the project).
    empty_branch : 'majority' (the paper's recommended fix) or 'null' (original ID3).
    """

    def __init__(self, criterion: str = "gain", chi2_alpha: Optional[float] = None,
                 max_depth: Optional[int] = None, min_samples_split: int = 2,
                 empty_branch: str = "majority", split: str = "multiway",
                 continuous: Optional[List[str]] = None, max_subset_values: int = 10):
        assert criterion in ("gain", "gain_ratio")
        assert empty_branch in ("majority", "null")
        assert split in ("multiway", "subset")
        self.criterion = criterion
        self.chi2_alpha = chi2_alpha
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.empty_branch = empty_branch
        self.split = split
        self.continuous = list(continuous or [])
        self.max_subset_values = max_subset_values

    # ------------------------------------------------------------------ fitting
    def fit(self, X: pd.DataFrame, y) -> "ID3":
        X = X.astype(object).reset_index(drop=True)
        y = pd.Series(np.asarray(y)).astype(str).reset_index(drop=True)
        self.classes_ = np.array(sorted(y.unique()))
        self.attributes_ = list(X.columns)
        self.domains_, self._X = {}, {}
        for a in X.columns:
            unk = _is_unknown(X[a])
            if a in self.continuous:
                col = pd.to_numeric(X[a].where(~unk), errors="coerce").to_numpy(dtype=float)
                self._X[a] = col                                   # NaN marks 'unknown'
                self.domains_[a] = None
            else:
                col = X[a].astype(str).to_numpy(dtype=object).copy()  # never write into the caller's frame
                col[unk] = None
                self._X[a] = col                                   # None marks 'unknown'
                # The domain of each attribute ("values that could have appeared")
                self.domains_[a] = sorted(set(col[~unk]))
        y_idx = np.searchsorted(self.classes_, y.to_numpy())
        self.root_ = self._build(np.arange(len(y_idx)), y_idx, list(X.columns), depth=0)
        del self._X
        return self

    def _counts(self, y_idx):
        return np.bincount(y_idx, minlength=len(self.classes_)).astype(float)

    def _score(self, table, unknown_counts):
        adjusted = _adjust(table, unknown_counts)
        g = information_gain(adjusted)
        if self.criterion == "gain":
            score = g
        else:
            iv = intrinsic_value(adjusted)
            score = g / iv if iv > 0 else 0.0
        return score, g, chi_square(table)[1]

    def _best_test(self, rows, y_idx, a):
        """Best test on attribute a: dict(score, gain, p, kind, param) or None."""
        K = len(self.classes_)
        col = self._X[a][rows]
        if a in self.continuous:
            known = ~np.isnan(col)
            unknown_counts = np.bincount(y_idx[~known], minlength=K).astype(float)
            x, c = col[known], y_idx[known]
            if len(x) < 2:
                return None
            order = np.argsort(x, kind="stable")
            x, c = x[order], c[order]
            onehot = np.eye(K)[c]
            left = np.cumsum(onehot, axis=0)[:-1]                  # counts with x <= cut
            total = left[-1] + onehot[-1]
            cuts = np.where(np.diff(x) > 0)[0]                      # between distinct values
            if len(cuts) == 0:
                return None
            # Vectorised over all candidate cuts: 2 x K tables, Section-6 adjustment, gain and IV
            L, R = left[cuts], total - left[cuts]
            nL, nR = L.sum(1, keepdims=True), R.sum(1, keepdims=True)
            n = nL + nR
            u = unknown_counts[None, :]
            La, Ra = L + nL / n * u, R + nR / n * u
            N = La.sum(1) + Ra.sum(1)

            def H(M):
                tot = M.sum(1, keepdims=True)
                q = np.where(M > 0, M / np.where(tot > 0, tot, 1), 1.0)
                return -(q * np.log2(q)).sum(1)

            I0 = entropy(total + unknown_counts)
            gains = I0 - (La.sum(1) / N * H(La) + Ra.sum(1) / N * H(Ra))
            if self.criterion == "gain":
                scores = gains
            else:
                pl = (nL / n).ravel()
                iv = -(pl * np.log2(pl) + (1 - pl) * np.log2(1 - pl))
                scores = np.where(iv > 0, gains / iv, 0.0)
            j = int(np.argmax(scores))
            i = cuts[j]
            p = chi_square(np.vstack([L[j], R[j]]))[1]
            best = dict(score=float(scores[j]), gain=float(gains[j]), p=p, kind="threshold",
                        param=(x[i] + x[i + 1]) / 2)
            return best

        known = np.array([v is not None for v in col])
        values = self.domains_[a]
        pos = {v: i for i, v in enumerate(values)}
        table = np.zeros((len(values), K))
        for v, c in zip(col[known], y_idx[known]):
            table[pos[v], c] += 1
        unknown_counts = np.bincount(y_idx[~known], minlength=K).astype(float)
        if table.sum() == 0:
            return None
        if self.split == "multiway":
            score, g, p = self._score(table, unknown_counts)
            return dict(score=score, gain=g, p=p, kind="value", param=None)

        present = [i for i in range(len(values)) if table[i].sum() > 0]
        if len(present) < 2:
            return None
        if len(present) <= self.max_subset_values:
            first, rest = present[0], present[1:]
            candidates = []
            for mask in range(2 ** len(rest)):                     # subsets that contain `first`
                S = [first] + [rest[j] for j in range(len(rest)) if mask >> j & 1]
                if len(S) < len(present):
                    candidates.append(S)
        else:
            frac = table[present, -1] / table[present].sum(axis=1)
            ordered = [present[i] for i in np.argsort(frac, kind="stable")]
            candidates = [ordered[:k] for k in range(1, len(ordered))]
        best = None
        for S in candidates:
            inside = table[S].sum(axis=0)
            t2 = np.vstack([inside, table.sum(axis=0) - inside])
            score, g, p = self._score(t2, unknown_counts)
            if best is None or score > best["score"] + 1e-12:
                best = dict(score=score, gain=g, p=p, kind="subset",
                            param=frozenset(values[i] for i in S))
        return best

    def _build(self, rows, y_idx_all, attributes, depth) -> Node:
        y_idx = y_idx_all[rows]
        dist = self._counts(y_idx)
        node = Node(dist=dist)
        if (dist > 0).sum() <= 1 or not attributes:
            return node                                    # pure, or attributes exhausted
        if self.max_depth is not None and depth >= self.max_depth:
            return node
        if len(rows) < self.min_samples_split:
            return node

        scored = []
        for a in attributes:
            t = self._best_test(rows, y_idx, a)
            if t is None or not np.isfinite(t["score"]):
                continue
            if self.chi2_alpha is not None and t["p"] >= self.chi2_alpha:
                continue                                   # Section 5: irrelevance not rejected
            scored.append((a, t))
        if not scored:
            return node                                    # majority-class leaf (Section 5)
        if self.criterion == "gain_ratio":
            # Section 7: maximise the ratio among attributes with at least average gain
            avg_gain = np.mean([t["gain"] for _, t in scored])
            scored = [s for s in scored if s[1]["gain"] >= avg_gain - 1e-12]
        best_a, best = max(scored, key=lambda s: (s[1]["score"], -self.attributes_.index(s[0])))
        if best["gain"] <= 1e-12:
            return node                                    # no split separates anything

        node.attribute, node.gain, node.p_value = best_a, best["gain"], best["p"]
        node.kind, node.param = best["kind"], best["param"]
        col = self._X[best_a][rows]
        if node.kind == "threshold":
            known = ~np.isnan(col)
            branches = {"le": known & (np.nan_to_num(col, nan=np.inf) <= node.param),
                        "gt": known & (np.nan_to_num(col, nan=-np.inf) > node.param)}
            remaining = attributes                         # a numeric attribute can be cut again
        else:
            known = np.array([v is not None for v in col])
            if node.kind == "value":
                branches = {v: known & (col == v) for v in self.domains_[best_a]}
                remaining = [a for a in attributes if a != best_a]
            else:
                inside = np.array([v in node.param for v in col])
                node._out = frozenset(self.domains_[best_a]) - node.param
                branches = {"in": known & inside, "out": known & ~inside}
                remaining = attributes                     # a subset test can be refined below
        n_known = known.sum()
        for key, m in branches.items():
            sub = rows[m]                                  # unknowns are discarded (Section 6)
            node.ratios[key] = len(sub) / n_known if n_known else 0.0
            if len(sub) == 0:
                node.children[key] = Node(dist=dist.copy(), is_null=(self.empty_branch == "null"))
            elif len(sub) == len(rows):
                node.children[key] = Node(dist=dist.copy())   # cannot happen for gain > 0; guard
            else:
                node.children[key] = self._build(sub, y_idx_all, remaining, depth + 1)
        return node

    # ------------------------------------------------------------------ prediction
    def _leaf_vote(self, node: Node, token: float) -> np.ndarray:
        if node.is_null:
            return np.zeros(len(self.classes_))            # 'null' fails to classify
        return token * node.dist / node.dist.sum()

    def _route(self, node: Node, row: dict, token: float) -> np.ndarray:
        if node.is_leaf:
            return self._leaf_vote(node, token)
        v = row.get(node.attribute)
        unknown = v is None or (isinstance(v, float) and np.isnan(v)) or str(v) in UNKNOWN_TOKENS
        key = None
        if not unknown:
            try:
                key = node.branch_of(v if node.kind == "threshold" else str(v))
            except (TypeError, ValueError):
                key = None
        if key is not None:
            return self._route(node.children[key], row, token)
        # Unknown (or never-seen) value: explore every branch with token * ratio_i (Section 6)
        total = np.zeros(len(self.classes_))
        for k, child in node.children.items():
            r = node.ratios.get(k, 0.0)
            if r > 0:
                total += self._route(child, row, token * r)
        return total

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        out = []
        for row in X.astype(object).to_dict(orient="records"):
            votes = self._route(self.root_, row, 1.0)
            s = votes.sum()
            out.append(votes / s if s > 0 else np.full(len(self.classes_), np.nan))
        return np.vstack(out)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        P = self.predict_proba(X)
        pred = np.empty(len(P), dtype=object)
        ok = ~np.isnan(P).any(axis=1)
        pred[ok] = self.classes_[P[ok].argmax(axis=1)]
        pred[~ok] = "null"                                  # unclassified by a 'null' leaf
        return pred

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y).astype(str)))

    # ------------------------------------------------------------------ inspection
    def _walk(self, node=None):
        node = node or self.root_
        yield node
        for c in node.children.values():
            yield from self._walk(c)

    @property
    def n_nodes(self) -> int:
        return sum(1 for _ in self._walk())

    @property
    def n_leaves(self) -> int:
        return sum(1 for n in self._walk() if n.is_leaf)

    @property
    def depth(self) -> int:
        def d(n):
            return 0 if n.is_leaf else 1 + max(d(c) for c in n.children.values())
        return d(self.root_)

    def export_text(self, node: Node = None, indent: str = "") -> str:
        node = node or self.root_
        lines = []
        if node.is_leaf:
            label = "null" if node.is_null else self.classes_[node.dist.argmax()]
            return f"{indent}-> {label}  " + ", ".join(f"{c}: {int(k)}" for c, k in zip(self.classes_, node.dist))
        for v, child in node.children.items():
            if child.is_leaf:
                lab = "null" if child.is_null else self.classes_[child.dist.argmax()]
                lines.append(f"{indent}{node.test_text(v)}: {lab}  (n={int(child.dist.sum())})")
            else:
                lines.append(f"{indent}{node.test_text(v)}:")
                lines.append(self.export_text(child, indent + "|   "))
        return "\n".join(lines)


# ----------------------------------------------------------------------------------------
# Windowing (Section 4)
# ----------------------------------------------------------------------------------------
def fit_windowed(model_factory, X: pd.DataFrame, y, window: int = 100, add: int = 50,
                 max_iter: int = 50, seed: int = 0):
    """Quinlan's iterative window: grow a tree on a random window, add misclassified objects
    until the tree is correct on the whole training set (or max_iter). Returns (model, history)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(str)
    n = len(y)
    in_window = np.zeros(n, dtype=bool)
    in_window[rng.choice(n, size=min(window, n), replace=False)] = True
    history = []
    for it in range(max_iter):
        model = model_factory().fit(X[in_window], y[in_window])
        wrong = np.where(model.predict(X) != y)[0]
        history.append({"iteration": it, "window": int(in_window.sum()), "errors": len(wrong)})
        if len(wrong) == 0:
            break
        new = wrong[~in_window[wrong]]
        if len(new) == 0:
            break
        in_window[rng.choice(new, size=min(add, len(new)), replace=False)] = True
    return model, history


# ----------------------------------------------------------------------------------------
# Corruption used in the paper's noise and ignorance experiments (Sections 5 and 6)
# ----------------------------------------------------------------------------------------
def corrupt(X: pd.DataFrame, y, level: float, where: str, domains: Dict[str, List[str]],
            classes, rng, attribute: Optional[str] = None):
    """With probability `level`, replace a value by one drawn uniformly from the values that
    could have appeared (it may coincide with the true value, as in the paper's footnote 5).

    where = 'class' | 'all' | 'single' (needs `attribute`)."""
    X = X.copy()
    y = np.asarray(y).astype(object).copy()
    n = len(y)
    if where == "class":
        hit = rng.random(n) < level
        y[hit] = rng.choice(np.asarray(classes, dtype=object), size=hit.sum())
    else:
        cols = [attribute] if where == "single" else list(X.columns)
        for a in cols:
            hit = rng.random(n) < level
            col = X[a].to_numpy(dtype=object).copy()
            col[hit] = rng.choice(np.asarray(domains[a], dtype=object), size=hit.sum())
            X[a] = col
    return X, y


def blank_out(X: pd.DataFrame, level: float, rng) -> pd.DataFrame:
    """Section 6 'ignorance level': each attribute value becomes unknown with probability level."""
    values = X.to_numpy(dtype=object).copy()
    values[rng.random(values.shape) < level] = "?"
    return pd.DataFrame(values, columns=X.columns, dtype=object)
