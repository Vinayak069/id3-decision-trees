"""
simulate.py -- simulated data with a KNOWN decision tree, so recovery can be checked exactly.

Ten categorical attributes A1..A10, each uniform on {a, b, c} and independent.
The class is a deterministic function of A1..A4 only (A5..A10 are irrelevant):

    A1 = a:  A2 = b -> N, otherwise P
    A1 = b:  P
    A1 = c:  A3 = c -> (A4 = a -> P, otherwise N), otherwise N

P(class = P) = 1/3*(2/3) + 1/3 + 1/3*(1/3*1/3) = 16/27 = 0.5926, so the majority-class
error rate is 11/27 = 0.4074, which is the ceiling against which noise results are read.

Optionally an 'id_code' attribute with many random values (irrelevant) is added to show the
gain criterion's bias towards many-valued attributes (paper Section 7).
"""
import numpy as np
import pandas as pd

VALUES = ["a", "b", "c"]
TRUE_TREE_TEXT = """A1 = a:
|   A2 = a: P
|   A2 = b: N
|   A2 = c: P
A1 = b: P
A1 = c:
|   A3 = a: N
|   A3 = b: N
|   A3 = c:
|   |   A4 = a: P
|   |   A4 = b: N
|   |   A4 = c: N"""
P_POSITIVE = 1 / 3 * 2 / 3 + 1 / 3 + 1 / 3 * 1 / 9


def true_class(row) -> str:
    if row["A1"] == "a":
        return "N" if row["A2"] == "b" else "P"
    if row["A1"] == "b":
        return "P"
    if row["A3"] == "c":
        return "P" if row["A4"] == "a" else "N"
    return "N"


def simulate(n: int = 2000, seed: int = 2026, n_attributes: int = 10, id_values: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f"A{j + 1}": rng.choice(VALUES, size=n) for j in range(n_attributes)})
    df["class"] = df.apply(true_class, axis=1)
    if id_values:
        df.insert(0, "id_code", [f"v{k:03d}" for k in rng.integers(0, id_values, size=n)])
    return df


if __name__ == "__main__":
    import os
    os.makedirs("data/simulated", exist_ok=True)
    simulate(2000, seed=2026).to_csv("data/simulated/sim_tree_2000.csv", index=False)
    simulate(600, seed=7, id_values=150).to_csv("data/simulated/sim_with_id_600.csv", index=False)
    print("wrote data/simulated/sim_tree_2000.csv and data/simulated/sim_with_id_600.csv")
