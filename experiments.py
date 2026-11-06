"""
experiments.py -- every table and figure in the project, reproducible from fixed seeds.

    python experiments.py            # full run, 20 repetitions per setting (about 10 minutes on 4 cores)
    python experiments.py --quick    # 5 repetitions, for a fast check

Real public datasets carry the experiments; the simulated data (known tree) are used only as
sanity checks. Outputs go to results/*.csv and figures/*.png.
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import id3

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
COLORS = {"single": "#2a7ab9", "all": "#d1495b", "class": "#3c9a5f", "id3": "#2a7ab9",
          "chi2": "#e08a1e", "cart": "#7a5195", "ratio": "#3c9a5f", "subset": "#d1495b"}
REPS = 20

# ---------------------------------------------------------------------------- data
# name: (file, class column, label, kind, numeric attributes?)
DATASETS = {
    "weather":  ("real/weather_quinlan1986.csv", "class", "Saturday mornings (paper Table 1)", "real", False),
    "chess":    ("real/chess_kr_vs_kp.csv", "class", "Chess endgame KR vs KP (UCI)", "real", False),
    "mushroom": ("real/mushrooms.csv", "type", "Mushroom (UCI)", "real", False),
    "votes":    ("real/house_votes_84.csv", "class", "Congressional votes 1984 (UCI)", "real", False),
    "car":      ("real/car_evaluation.csv", "class", "Car Evaluation (UCI)", "real", False),
    "tictactoe": ("real/tic_tac_toe.csv", "class", "Tic-tac-toe endgame (UCI)", "real", False),
    "bcl":      ("real/breast_cancer_ljubljana.csv", "class", "Breast cancer, Ljubljana (UCI)", "real", False),
    "wdbc":     ("real/breast_cancer_wisconsin.csv", "class", "Breast cancer, Wisconsin (UCI)", "real", True),
    "sim":      ("simulated/sim_tree_2000.csv", "class", "Simulated: known depth-3 tree", "simulated", False),
    "sim_id":   ("simulated/sim_with_id_600.csv", "class", "Simulated + 150-valued id_code", "simulated", False),
}
BENCH = ["chess", "mushroom", "votes", "car", "tictactoe", "bcl", "wdbc"]
SHORT = {"chess": "Chess KRvKP", "mushroom": "Mushroom", "votes": "Votes 1984", "car": "Car", "tictactoe": "Tic-tac-toe",
         "bcl": "Breast c. (Ljubljana)", "wdbc": "Breast c. (Wisconsin)", "sim": "Simulated"}


def load(name: str):
    f, ycol, _, _, numeric = DATASETS[name]
    d = pd.read_csv(os.path.join(HERE, "data", f), dtype=None if numeric else str, keep_default_na=False)
    X = d.drop(columns=[c for c in (ycol, "no") if c in d.columns])
    return X, d[ycol].astype(str).to_numpy()


def is_numeric(name):
    return DATASETS[name][4]


def split(X, y, test_frac, rng):
    """Stratified random split."""
    idx_tr, idx_te = [], []
    for c in np.unique(y):
        ids = rng.permutation(np.where(y == c)[0])
        k = int(round(len(ids) * test_frac))
        idx_te += list(ids[:k]); idx_tr += list(ids[k:])
    idx_tr, idx_te = np.array(idx_tr), np.array(idx_te)
    return (X.iloc[idx_tr].reset_index(drop=True), y[idx_tr],
            X.iloc[idx_te].reset_index(drop=True), y[idx_te])


def se(a):
    a = np.asarray(a, dtype=float)
    return a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0


def corrected_t(diff, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for J random train/test splits:
    t = mean(d) / sqrt((1/J + n_test/n_train) * var(d)), J-1 degrees of freedom.
    The correction accounts for the overlap of training sets between splits."""
    d = np.asarray(diff, dtype=float)
    J = len(d)
    v = d.var(ddof=1)
    if v == 0:
        return (0.0, 1.0) if d.mean() == 0 else (float(np.sign(d.mean()) * np.inf), 0.0)
    t = d.mean() / np.sqrt((1 / J + n_test / n_train) * v)
    return float(t), float(2 * stats.t.sf(abs(t), J - 1))


def make(name, **kw):
    if is_numeric(name):
        X, _ = load(name)
        kw["continuous"] = list(X.columns)
    return id3.ID3(**kw)


# ---------------------------------------------------------------------------- E1
def e1_weather():
    """Reproduce the worked example of Sections 4 and 7 on Table 1."""
    X, y = load("weather")
    paper = {"outlook": (0.246, 1.578, 0.156), "temperature": (0.029, None, None),
             "humidity": (0.151, 1.000, 0.151), "windy": (0.048, None, None)}
    rows = []
    for a in X.columns:
        t = pd.crosstab(X[a], y).to_numpy()
        stat, p = id3.chi_square(t)
        rows.append({"attribute": a, "E(A) bits": id3.expected_information(t),
                     "gain bits": id3.information_gain(t), "paper gain": paper[a][0],
                     "IV bits": id3.intrinsic_value(t), "paper IV": paper[a][1],
                     "gain ratio": id3.gain_ratio(t), "paper gain ratio": paper[a][2],
                     "chi2": stat, "p-value": p})
    df = pd.DataFrame(rows).round(4)
    df.to_csv(os.path.join(RES, "e1_weather_gains.csv"), index=False)
    tree = id3.ID3().fit(X, y)
    tree_chi = id3.ID3(chi2_alpha=0.01).fit(X, y)
    with open(os.path.join(RES, "e1_weather_tree.txt"), "w") as f:
        f.write(f"I(9,5) = {id3.entropy([9, 5]):.4f} bits (paper 0.940)\n\n")
        f.write("ID3 tree (gain):\n" + tree.export_text() + "\n\n")
        f.write(f"Training accuracy {tree.score(X, y):.3f}; nodes {tree.n_nodes}, leaves {tree.n_leaves}\n\n")
        f.write("With the 99% chi-square test the tree is a single leaf:\n" + tree_chi.export_text() +
                "\n(14 objects are too few to reject independence)\n")
    return df, tree


# ---------------------------------------------------------------------------- E2 benchmark
MODELS = {
    "ID3 (gain)": dict(),
    "ID3 (gain ratio)": dict(criterion="gain_ratio"),
    "ID3 + chi2 99%": dict(chi2_alpha=0.01),
    "ID3 subsets (Sec. 7)": dict(split="subset"),
    "ID3 subsets + chi2 99%": dict(split="subset", chi2_alpha=0.01),
}


def _cart(name, Xtr, ytr, Xte, yte):
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(criterion="entropy", random_state=0)
    if is_numeric(name):
        clf.fit(Xtr, ytr)
        return clf.score(Xte, yte), clf.tree_.node_count
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(handle_unknown="ignore")          # '?' becomes its own indicator
    clf.fit(enc.fit_transform(Xtr), ytr)
    return clf.score(enc.transform(Xte), yte), clf.tree_.node_count


def _e2_one(args):
    name, seed, train_frac = args
    X, y = load(name)
    Xtr, ytr, Xte, yte = split(X, y, 1 - train_frac, np.random.default_rng(seed))
    out = []
    for m, kw in MODELS.items():
        if is_numeric(name) and kw.get("split") == "subset":
            continue                                          # thresholds are already binary
        t0 = time.time(); tree = make(name, **kw).fit(Xtr, ytr)
        out.append(dict(dataset=name, seed=seed, train_frac=train_frac, model=m, accuracy=tree.score(Xte, yte),
                        nodes=tree.n_nodes, depth=tree.depth, fit_s=time.time() - t0, n_train=len(ytr), n_test=len(yte)))
    acc, nodes = _cart(name, Xtr, ytr, Xte, yte)
    out.append(dict(dataset=name, seed=seed, train_frac=train_frac, model="CART (sklearn)", accuracy=acc,
                    nodes=nodes, depth=np.nan, fit_s=np.nan, n_train=len(ytr), n_test=len(yte)))
    maj = pd.Series(ytr).value_counts().idxmax()
    out.append(dict(dataset=name, seed=seed, train_frac=train_frac, model="Majority class",
                    accuracy=float(np.mean(yte == maj)), nodes=1, depth=0, fit_s=0, n_train=len(ytr), n_test=len(yte)))
    return out


def e2_benchmark(pool, reps):
    """Seven real datasets, 70 % training, `reps` stratified splits; corrected paired t-tests."""
    jobs = [(n, s, 0.7) for n in BENCH for s in range(reps)]
    df = pd.DataFrame([r for res in pool.map(_e2_one, jobs) for r in res])
    df.to_csv(os.path.join(RES, "e2_benchmark_raw.csv"), index=False)
    summ = df.groupby(["dataset", "model"], sort=False).agg(
        acc_mean=("accuracy", "mean"), acc_se=("accuracy", se), nodes=("nodes", "mean")).reset_index()
    summ.round(4).to_csv(os.path.join(RES, "e2_benchmark_summary.csv"), index=False)

    tests = []
    for name in BENCH:
        d = df[df.dataset == name].pivot(index="seed", columns="model", values="accuracy")
        ntr, nte = df[df.dataset == name].n_train.iloc[0], df[df.dataset == name].n_test.iloc[0]
        pairs = [("ID3 (gain)", "CART (sklearn)"), ("ID3 + chi2 99%", "ID3 (gain)"), ("ID3 (gain ratio)", "ID3 (gain)")]
        if not is_numeric(name):
            pairs = [("ID3 subsets (Sec. 7)", "ID3 (gain)"), ("ID3 subsets (Sec. 7)", "CART (sklearn)")] + pairs
        for a, b in pairs:
            diff = d[a] - d[b]
            t, p = corrected_t(diff, ntr, nte)
            tests.append(dict(dataset=name, comparison=f"{a} - {b}", mean_diff_pts=100 * diff.mean(),
                              wins=int((diff > 0).sum()), ties=int((diff == 0).sum()), losses=int((diff < 0).sum()),
                              t=t, p_value=p))
    tests = pd.DataFrame(tests).round(4)
    tests.to_csv(os.path.join(RES, "e2_paired_tests.csv"), index=False)

    order = ["ID3 (gain)", "ID3 (gain ratio)", "ID3 + chi2 99%", "ID3 subsets (Sec. 7)", "CART (sklearn)"]
    col = {"ID3 (gain)": COLORS["id3"], "ID3 (gain ratio)": COLORS["ratio"], "ID3 + chi2 99%": COLORS["chi2"],
           "ID3 subsets (Sec. 7)": COLORS["subset"], "CART (sklearn)": COLORS["cart"]}
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for i, name in enumerate(BENCH):
        for j, m in enumerate(order):
            r = summ[(summ.dataset == name) & (summ.model == m)]
            if len(r) == 0:
                continue
            ax.errorbar(100 * r.acc_mean, i + (j - 2) * 0.13, xerr=196 * r.acc_se, fmt="o", ms=5,
                        color=col[m], label=m if i == 0 else None)
        maj = summ[(summ.dataset == name) & (summ.model == "Majority class")].acc_mean.iloc[0]
        ax.plot(100 * maj, i, "|", color="#999", ms=14, label="majority class" if i == 0 else None)
    ax.set_yticks(range(len(BENCH))); ax.set_yticklabels([SHORT[n] for n in BENCH]); ax.invert_yaxis()
    ax.set_xlabel("test accuracy (%), mean and 95% CI over 20 splits (70% training)"); ax.set_xlim(55, 101)
    ax.legend(fontsize=8, loc="lower left", ncol=3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "e2_benchmark.png")); plt.close(fig)
    return summ, tests


def _e2b_one(args):
    frac, seed = args
    X, y = load("chess")
    Xtr, ytr, Xte, yte = split(X, y, 1 - frac, np.random.default_rng(700 + seed))
    t = id3.ID3().fit(Xtr, ytr)
    return dict(train_frac=frac, seed=seed, accuracy=t.score(Xte, yte), nodes=t.n_nodes)


def e2b_learning_curve(pool, reps):
    """Section 4's experiment: train on a random fraction of the chess positions, test on the rest."""
    jobs = [(f, s) for f in (0.02, 0.05, 0.1, 0.2, 0.4, 0.7) for s in range(reps)]
    df = pd.DataFrame(list(pool.map(_e2b_one, jobs)))
    summ = df.groupby("train_frac").agg(accuracy=("accuracy", "mean"), acc_se=("accuracy", se),
                                        nodes=("nodes", "mean")).reset_index().round(4)
    summ.to_csv(os.path.join(RES, "e2b_chess_learning_curve.csv"), index=False)
    return summ


# ---------------------------------------------------------------------------- E3 depth + alpha
DEPTH_SETS = ["chess", "car", "votes", "bcl"]
ALPHAS = [None, 0.5, 0.2, 0.1, 0.05, 0.01, 0.001, 0.0001]


def _e3_one(args):
    name, seed = args
    X, y = load(name)
    Xtr, ytr, Xte, yte = split(X, y, 0.3, np.random.default_rng(2000 + seed))
    out = []
    for d in list(range(0, 9)) + [None]:
        t = make(name, max_depth=d).fit(Xtr, ytr)
        out.append(dict(dataset=name, seed=seed, control="depth", value=99 if d is None else d,
                        train_acc=t.score(Xtr, ytr), test_acc=t.score(Xte, yte), nodes=t.n_nodes))
    for a in ALPHAS:
        t = make(name, chi2_alpha=a).fit(Xtr, ytr)
        out.append(dict(dataset=name, seed=seed, control="alpha", value=1.0 if a is None else a,
                        train_acc=t.score(Xtr, ytr), test_acc=t.score(Xte, yte), nodes=t.n_nodes))
    return out


def e3_depth_alpha(pool, reps, sets=DEPTH_SETS, tag="real"):
    """Depth control and the chi-square confidence level."""
    jobs = [(n, s) for n in sets for s in range(reps)]
    df = pd.DataFrame([r for res in pool.map(_e3_one, jobs) for r in res])
    summ = df.groupby(["dataset", "control", "value"]).agg(
        train_acc=("train_acc", "mean"), test_acc=("test_acc", "mean"), test_se=("test_acc", se),
        nodes=("nodes", "mean")).reset_index().round(4)
    summ.to_csv(os.path.join(RES, f"e3_depth_alpha_{tag}.csv"), index=False)
    plot_e3(summ, sets, tag)
    return summ


def plot_e3(summ, sets, tag):
    fig, axes = plt.subplots(2, len(sets), figsize=(3.6 * max(len(sets), 2), 6.2), squeeze=False)
    for k, name in enumerate(sets):
        s = summ[(summ.dataset == name) & (summ.control == "depth")]
        ax = axes[0, k]; x = s.value.replace(99, 9)
        ax.plot(x, 100 * s.train_acc, "o-", ms=3, color="#999", label="train")
        ax.errorbar(x, 100 * s.test_acc, yerr=196 * s.test_se, fmt="o-", ms=3, color=COLORS["id3"], label="test")
        best = s.loc[s.test_acc.idxmax()]
        ax.plot(min(best.value, 9), 100 * best.test_acc, "o", ms=9, mfc="none", mec=COLORS["subset"], mew=1.5)
        ax.text(0.97, 0.45, f"best test: {'depth ' + str(int(best.value)) if best.value < 99 else 'full tree'}\n"
                f"({best.nodes:.0f} node{'' if round(best.nodes) == 1 else 's'})", transform=ax.transAxes, ha="right", va="center", fontsize=8,
                color=COLORS["subset"], bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85))
        ax.set_xticks(range(10)); ax.set_xticklabels([str(i) for i in range(9)] + ["full"], fontsize=8)
        ax.set_title(SHORT.get(name, name)); ax.set_xlabel("max depth")
        s = summ[(summ.dataset == name) & (summ.control == "alpha")].sort_values("value", ascending=False)
        ax = axes[1, k]; xs = np.arange(len(s))
        ax2 = ax.twinx(); ax2.bar(xs, s.nodes, color="#cccccc", alpha=0.6, width=0.5)
        ax2.set_ylabel("nodes (bars)", fontsize=8, color="#888"); ax2.tick_params(labelsize=7, colors="#888"); ax2.grid(False)
        ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
        ax.errorbar(xs, 100 * s.test_acc, yerr=196 * s.test_se, fmt="o-", ms=3, color=COLORS["chi2"])
        ax.set_xticks(xs); ax.set_xticklabels(["none" if v == 1 else f"{v:g}" for v in s.value], rotation=45, fontsize=7)
        ax.set_xlabel("χ² level α (split only if p < α)")
    axes[0, 0].set_ylabel("accuracy (%)"); axes[1, 0].set_ylabel("test accuracy (%)"); axes[0, 0].legend(fontsize=8)
    for k in range(len(sets), axes.shape[1]):
        axes[0, k].axis("off"); axes[1, k].axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"e3_depth_alpha_{tag}.png")); plt.close(fig)


# ---------------------------------------------------------------------------- E4 noise
NOISE_LEVELS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
PAPER_T2 = pd.DataFrame({"level": [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                         "single": [1.3, 2.5, 3.3, 4.6, 6.1, 7.6, 8.8, 9.4, 9.9, 10.4, 10.8, 10.8],
                         "all": [11.9, 18.9, 24.6, 27.8, 29.5, 30.3, 29.2, 27.5, 25.9, 26.0, 25.6, 25.9],
                         "class": [2.6, 5.5, 8.3, 9.9, 14.8, 18.1, 21.8, 26.4, 27.2, 29.5, 34.1, 49.6]})


def root_attribute(name):
    X, y = load(name)
    return id3.ID3(max_depth=1).fit(X, y).root_.attribute


def _e4_one(args):
    name, where, level, alpha, seed, single_attr = args
    X, y = load(name)
    rng = np.random.default_rng(1000 + seed)
    Xtr, ytr, Xte, yte = split(X, y, 0.3, rng)
    dom = {a: sorted(X[a].unique()) for a in X.columns}
    cls = np.unique(y)
    Xtr_c, ytr_c = id3.corrupt(Xtr, ytr, level, where, dom, cls, rng, attribute=single_attr)
    Xte_c, yte_c = id3.corrupt(Xte, yte, level, where, dom, cls, rng, attribute=single_attr)
    tree = id3.ID3(chi2_alpha=alpha).fit(Xtr_c, ytr_c)
    row = dict(dataset=name, where=where, level=level, chi2=alpha is not None, seed=seed,
               error=1 - tree.score(Xte_c, yte_c), nodes=tree.n_nodes, error_clean_tree=np.nan)
    if where == "all" and alpha is not None:
        clean = id3.ID3(chi2_alpha=alpha).fit(Xtr, ytr)          # the paper's "laboratory" tree
        row["error_clean_tree"] = 1 - clean.score(Xte_c, yte_c)
    return row


def e4_noise(pool, reps, sets=("chess", "car"), tag="real"):
    """Replicates Table 2: noise in the root attribute, in all attributes, and in the class."""
    single = {n: root_attribute(n) for n in sets}
    jobs = [(n, w, l, a, s, single[n]) for n in sets for w in ("single", "all", "class")
            for l in NOISE_LEVELS for a in (0.01, None) for s in range(reps)]
    df = pd.DataFrame(list(pool.map(_e4_one, jobs, chunksize=4)))
    summ = df.groupby(["dataset", "where", "chi2", "level"]).agg(
        error=("error", "mean"), error_se=("error", se), nodes=("nodes", "mean"),
        error_clean_tree=("error_clean_tree", "mean")).reset_index().round(4)
    summ["single_attribute"] = summ.dataset.map(single)
    summ.to_csv(os.path.join(RES, f"e4_noise_{tag}.csv"), index=False)
    summ[summ.chi2].pivot_table(index="level", columns=["dataset", "where"], values="error") \
        .to_csv(os.path.join(RES, f"e4_table2_{tag}.csv"))
    PAPER_T2.to_csv(os.path.join(RES, "paper_table2.csv"), index=False)

    fig, axes = plt.subplots(1, len(sets) + 1, figsize=(4.7 * (len(sets) + 1), 3.8))
    for ax, name in zip(axes, sets):
        for w in ("single", "all", "class"):
            s = summ[(summ.dataset == name) & (summ["where"] == w) & summ.chi2]
            ax.errorbar(s.level * 100, s.error * 100, yerr=196 * s.error_se, fmt="o-", ms=3, color=COLORS[w],
                        label={"single": f"one attribute ({single[name]})", "all": "all attributes", "class": "class"}[w])
        ax.set_title(f"{SHORT[name]}: ID3 + χ² 99%"); ax.set_xlabel("noise level (%)"); ax.set_ylabel("test error (%)")
        ax.legend(fontsize=8)
    for w in ("single", "all", "class"):
        axes[-1].plot(PAPER_T2.level, PAPER_T2[w], "o-", ms=3, color=COLORS[w], label=w)
    axes[-1].set_title("Quinlan (1986) Table 2 (chess, 551 objects)"); axes[-1].set_xlabel("noise level (%)")
    axes[-1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"e4_noise_{tag}.png")); plt.close(fig)

    name = sets[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, w in zip(axes, ("all", "class")):
        for chi, c, lab in ((True, COLORS["chi2"], "with χ² test (99%)"), (False, COLORS["id3"], "no test")):
            s = summ[(summ.dataset == name) & (summ["where"] == w) & (summ.chi2 == chi)]
            ax.plot(s.level * 100, s.error * 100, "o-", ms=3, color=c, label=lab)
        if w == "all":
            s = summ[(summ.dataset == name) & (summ["where"] == "all") & summ.chi2]
            ax.plot(s.level * 100, s.error_clean_tree * 100, "s--", ms=3, color="#555", label="tree built on clean data")
        ax.set_title(f"{SHORT[name]}: noise in {'all attributes' if w == 'all' else 'the class'}")
        ax.set_xlabel("noise level (%)"); ax.set_ylabel("test error (%)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"e4_chi2_ablation_{tag}.png")); plt.close(fig)
    return summ


# ---------------------------------------------------------------------------- E5 unknowns
IGNORANCE = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def mode_impute(Xtr, Xte):
    """The paper's 'simple strategy': replace an unknown by the attribute's most common value."""
    Xtr, Xte = Xtr.copy().astype(object), Xte.copy().astype(object)
    for a in Xtr.columns:
        known = Xtr[a][Xtr[a] != "?"]
        m = known.mode().iloc[0] if len(known) else "?"
        Xtr[a] = Xtr[a].replace("?", m); Xte[a] = Xte[a].replace("?", m)
    return Xtr, Xte


def as_value(D):
    v = D.to_numpy(dtype=object)
    return pd.DataFrame(np.where(v == "?", "UNK", v), columns=D.columns, dtype=object)


def _three_ways(Xtr, ytr, Xte, yte):
    out = {"Quinlan Sec. 6": id3.ID3().fit(Xtr, ytr).score(Xte, yte),
           "unknown as a value": id3.ID3().fit(as_value(Xtr), ytr).score(as_value(Xte), yte)}
    a, b = mode_impute(Xtr, Xte)
    out["most common value"] = id3.ID3().fit(a, ytr).score(b, yte)
    return out


def _e5_one(args):
    name, level, seed = args
    X, y = load(name)
    rng = np.random.default_rng(5000 + seed)
    Xtr, ytr, Xte, yte = split(X, y, 0.3, rng)
    Xtr_u, Xte_u = id3.blank_out(Xtr, level, rng), id3.blank_out(Xte, level, rng)
    out = dict(dataset=name, level=level, seed=seed)
    for k, v in _three_ways(Xtr_u, ytr, Xte_u, yte).items():
        out["error: " + k] = 1 - v
    out["error: clean tree, incomplete objects"] = 1 - id3.ID3().fit(Xtr, ytr).score(Xte_u, yte)
    return out


def _e5_real(args):
    name, seed = args
    X, y = load(name)
    Xtr, ytr, Xte, yte = split(X, y, 0.3, np.random.default_rng(9000 + seed))
    return dict(dataset=name, seed=seed, **_three_ways(Xtr, ytr, Xte, yte))


def e5_unknown(pool, reps, sets=("chess", "car"), tag="real"):
    """Figure 5 (ignorance level) on real data, and three strategies on genuinely missing data."""
    jobs = [(n, l, s) for n in sets for l in IGNORANCE for s in range(reps)]
    df = pd.DataFrame(list(pool.map(_e5_one, jobs, chunksize=4)))
    summ = df.groupby(["dataset", "level"]).mean(numeric_only=True).drop(columns="seed").reset_index().round(4)
    summ.to_csv(os.path.join(RES, f"e5_ignorance_{tag}.csv"), index=False)

    real = pd.DataFrame(list(pool.map(_e5_real, [(n, s) for n in ("votes", "mushroom", "bcl") for s in range(reps)])))
    rs = real.drop(columns="seed").groupby("dataset", sort=False).agg(["mean", se])
    rs.columns = [f"{a} ({'mean' if b == 'mean' else 'se'})" for a, b in rs.columns]
    rs = rs.reset_index().round(4)
    rs.to_csv(os.path.join(RES, "e5_real_missing.csv"), index=False)

    fig, axes = plt.subplots(1, len(sets) + 1, figsize=(4.6 * (len(sets) + 1), 3.6))
    styles = {"error: Quinlan Sec. 6": ("o-", COLORS["id3"]), "error: unknown as a value": ("o-", COLORS["all"]),
              "error: most common value": ("^-", COLORS["chi2"]), "error: clean tree, incomplete objects": ("s--", "#555")}
    for ax, name in zip(axes, sets):
        s = summ[summ.dataset == name]
        for c, (fmt, colr) in styles.items():
            ax.plot(s.level * 100, s[c] * 100, fmt, ms=3, color=colr, label=c.replace("error: ", ""))
        ax.set_title(f"{SHORT[name]}: unknown values"); ax.set_xlabel("ignorance level (%)"); ax.set_ylabel("test error (%)")
    axes[0].legend(fontsize=7)
    ax = axes[-1]
    names = list(rs.dataset); xs = np.arange(len(names))
    for j, (k, colr) in enumerate([("Quinlan Sec. 6", COLORS["id3"]), ("unknown as a value", COLORS["all"]),
                                   ("most common value", COLORS["chi2"])]):
        ax.bar(xs + (j - 1) * 0.26, 100 * (1 - rs[f"{k} (mean)"]), width=0.26, yerr=196 * rs[f"{k} (se)"],
               color=colr, label=k)
    ax.set_xticks(xs); ax.set_xticklabels([SHORT[n] for n in names], fontsize=8)
    ax.set_ylabel("test error (%)"); ax.set_title("Genuinely missing values (real data)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"e5_unknown_{tag}.png")); plt.close(fig)
    return summ, rs


# ---------------------------------------------------------------------------- E6 gain bias
def e6_gain_bias(reps):
    """Section 7: a many-valued irrelevant attribute fools the gain criterion, not the gain ratio
    (simulated sanity check, because the truth is known)."""
    X, y = load("sim_id")
    rows = []
    for s in range(reps):
        Xtr, ytr, Xte, yte = split(X, y, 0.3, np.random.default_rng(300 + s))
        for crit, kw in (("gain", {}), ("gain ratio", {"criterion": "gain_ratio"}), ("gain, binary subsets", {"split": "subset"})):
            t = id3.ID3(**kw).fit(Xtr, ytr)
            uses = sum(1 for n in t._walk() if n.attribute == "id_code")
            rows.append(dict(seed=s, criterion=crit, root=t.root_.attribute, id_code_tests=uses,
                             accuracy=t.score(Xte, yte), nodes=t.n_nodes))
    df = pd.DataFrame(rows)
    summ = df.groupby("criterion", sort=False).agg(
        id_code_tests=("id_code_tests", "mean"), accuracy=("accuracy", "mean"), acc_se=("accuracy", se),
        nodes=("nodes", "mean")).reset_index().round(4)
    summ.to_csv(os.path.join(RES, "e6_gain_bias.csv"), index=False)
    Xtr, ytr, _, _ = split(X, y, 0.3, np.random.default_rng(300))
    g = []
    for a in X.columns:
        t = pd.crosstab(Xtr[a], ytr).to_numpy()
        g.append(dict(attribute=a, values=Xtr[a].nunique(), gain=id3.information_gain(t),
                      IV=id3.intrinsic_value(t), gain_ratio=id3.gain_ratio(t)))
    pd.DataFrame(g).sort_values("gain", ascending=False).round(4).to_csv(os.path.join(RES, "e6_root_scores.csv"), index=False)
    return summ


# ---------------------------------------------------------------------------- E7 windowing
def e7_windowing():
    """Section 4: the window, on the chess endgame and mushroom data."""
    rows, hists = [], {}
    for name in ("chess", "mushroom"):
        X, y = load(name)
        Xtr, ytr, Xte, yte = split(X, y, 0.3, np.random.default_rng(11))
        t0 = time.time(); direct = id3.ID3().fit(Xtr, ytr); td = time.time() - t0
        t0 = time.time(); win, hist = id3.fit_windowed(lambda: id3.ID3(), Xtr, ytr, window=100, add=50, seed=0); tw = time.time() - t0
        hists[name] = hist
        rows += [dict(dataset=name, method="direct", objects_used=len(ytr), nodes=direct.n_nodes,
                      train_errors=int((direct.predict(Xtr) != ytr).sum()), test_accuracy=direct.score(Xte, yte), seconds=td),
                 dict(dataset=name, method="windowing (100 + 50 per round)", objects_used=hist[-1]["window"],
                      nodes=win.n_nodes, train_errors=hist[-1]["errors"], test_accuracy=win.score(Xte, yte), seconds=tw)]
    out = pd.DataFrame(rows).round(4)
    out.to_csv(os.path.join(RES, "e7_windowing.csv"), index=False)
    pd.concat([pd.DataFrame(h).assign(dataset=n) for n, h in hists.items()]).to_csv(
        os.path.join(RES, "e7_windowing_history.csv"), index=False)
    return out


# ---------------------------------------------------------------------------- E8 theory
def e8_theory():
    """Section 5's explanation of the error peak: probabilistic leaf vs majority leaf."""
    q = np.linspace(0.5, 1.0, 51)
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.plot(q, 2 * q * (1 - q), color=COLORS["all"], label="classify P with prob. p/(p+n):  2pn/(p+n)²")
    ax.plot(q, 1 - q, color=COLORS["id3"], label="always the majority class:  n/(p+n)")
    ax.set_xlabel("p/(p+n)"); ax.set_ylabel("expected error"); ax.legend(fontsize=7)
    ax.set_title("Why error falls again at high noise")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "e8_theory_peak.png")); plt.close(fig)


# ---------------------------------------------------------------------------- sanity checks
def sanity_checks(pool, reps):
    """Simulated data, where the truth is known: recovery of the generating tree, depth and noise."""
    from simulate import TRUE_TREE_TEXT
    X, y = load("sim")
    t = id3.ID3().fit(X, y)
    recovered = "\n".join(l.split("  (n=")[0] for l in t.export_text().split("\n"))
    ok = recovered.strip() == TRUE_TREE_TEXT.strip()
    with open(os.path.join(RES, "s1_sim_recovery.txt"), "w") as f:
        f.write(f"ID3 on 2000 simulated objects recovers the generating tree exactly: {ok}\n\n{t.export_text()}\n")
    e3_depth_alpha(pool, reps, sets=["sim"], tag="sim")
    e4_noise(pool, reps, sets=("sim",), tag="sim")
    return ok


# ---------------------------------------------------------------------------- tree figure
def draw_weather_tree(tree, path=os.path.join(FIG, "weather_tree.png")):
    """Hand-drawn layout of the Figure-2 tree learned from Table 1."""
    fig, ax = plt.subplots(figsize=(7, 3.6)); ax.axis("off")
    box = dict(boxstyle="round,pad=0.4", fc="#e8f0f8", ec="#2a7ab9")
    leafP = dict(boxstyle="round,pad=0.3", fc="#e6f4ea", ec="#3c9a5f")
    leafN = dict(boxstyle="round,pad=0.3", fc="#fbe9eb", ec="#d1495b")

    def text(x, yv, s, b):
        ax.text(x, yv, s, ha="center", va="center", bbox=b, fontsize=10)

    def edge(x0, y0, x1, y1, lab):
        ax.plot([x0, x1], [y0 - 0.06, y1 + 0.06], color="#777", lw=1)
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, lab, fontsize=8, ha="center", va="center",
                color="#333", bbox=dict(fc="white", ec="none", pad=0.5))

    root = tree.root_
    text(0.5, 0.9, f"outlook\ngain {root.gain:.3f}", box)
    for v, x in {"sunny": 0.17, "overcast": 0.5, "rain": 0.83}.items():
        child = root.children[v]
        edge(0.5, 0.9, x, 0.55, v)
        if child.is_leaf:
            lab = tree.classes_[child.dist.argmax()]
            text(x, 0.55, f"{lab}  ({int(child.dist.sum())})", leafP if lab == "P" else leafN)
        else:
            text(x, 0.55, f"{child.attribute}\ngain {child.gain:.3f}", box)
            for k, (val, g) in enumerate(child.children.items()):
                xx = x + (-0.11 if k == 0 else 0.11)
                edge(x, 0.55, xx, 0.15, val)
                lab = tree.classes_[g.dist.argmax()]
                text(xx, 0.15, f"{lab}  ({int(g.dist.sum())})", leafP if lab == "P" else leafN)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ---------------------------------------------------------------------------- data summary
def data_summary():
    rows = []
    for name, (f, ycol, label, kind, numeric) in DATASETS.items():
        X, y = load(name)
        vc = pd.Series(y).value_counts()
        n_unknown = int(sum((X[a].astype(str) == "?").sum() for a in X.columns))
        nvals = "continuous" if numeric else f"{min(X[a].nunique() for a in X.columns)}-{max(X[a].nunique() for a in X.columns)}"
        rows.append({"dataset": label, "file": f, "type": kind, "objects": len(y), "attributes": X.shape[1],
                     "values per attribute": nvals, "classes": ", ".join(f"{k} {v}" for k, v in vc.items()),
                     "unknown values": n_unknown})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "data_summary.csv"), index=False)
    return df


def main(quick=False):
    reps = 5 if quick else REPS
    t0 = time.time()
    lap = lambda s: print(f"[{time.time() - t0:5.0f}s] {s}", flush=True)
    print(data_summary().to_string(index=False))
    df1, tree = e1_weather(); draw_weather_tree(tree); print(df1.to_string(index=False))
    e8_theory()
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        summ, tests = e2_benchmark(pool, reps); lap("E2 benchmark")
        print(summ.pivot(index="model", columns="dataset", values="acc_mean").round(4).to_string())
        print(tests.to_string(index=False))
        print(e2b_learning_curve(pool, reps).to_string(index=False)); lap("E2b learning curve")
        e3_depth_alpha(pool, reps); lap("E3 depth/alpha")
        e4_noise(pool, reps); lap("E4 noise")
        e5_unknown(pool, reps); lap("E5 unknown")
        print("sanity: simulated tree recovered =", sanity_checks(pool, reps)); lap("sanity checks")
    print(e6_gain_bias(reps).to_string(index=False))
    print(e7_windowing().to_string(index=False)); lap("all done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
