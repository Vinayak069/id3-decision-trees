# ID3 decision trees: Quinlan (1986), implemented from scratch

Advanced Machine Learning project (MA30235 / MA60215, IIT Kharagpur, Autumn 2026), paper 23 on the reading list: *Implement entropy and information gain; test depth control and noisy inputs.*

> Quinlan, J. R. (1986). Induction of decision trees. *Machine Learning*, 1(1), 81–106. https://doi.org/10.1007/BF00116251

Repository: https://github.com/Vinayak069/id3-decision-trees

**Group:** Vinayak Singh (22MA25005), Debjit Mondal (22MA25009), Keya Battu (22MA25006), Gurkirat Singh (22MA25015), Nikhil Nagar (22MA25001), Mayank Fenin (22MA25002), Atharv Sable (21MA20001).

## Results at a glance

All numbers are means over 20 stratified 70/30 splits (see `results/`).

* **The code reproduces the paper.** On Table 1 we get I(9,5) = 0.940 bits and gain(outlook) = 0.246 bits, the same as the paper. The tree is the paper's Figure 2. On simulated data ID3 recovers the generating tree exactly.
* **Benchmark on 7 real UCI datasets.** Plain multi-way ID3 trails CART by 4–9 points on Car Evaluation and Tic-tac-toe, because multi-way splits fragment the data. The paper's own binary subset tests (Section 7) close the gap: 97.4 % vs CART's 96.8 % on Car, and 92.9 % vs 93.2 % on Tic-tac-toe. Elsewhere all variants are within the noise.
* **Depth control.** Clean concepts (chess, car) need deep trees. On noisy data shallow trees win: Votes 1984 is best at depth 1 (96.1 % vs 94.4 % for the full tree), and on Ljubljana breast cancer nothing beats the majority class.
* **Noisy inputs.** On real chess and car data the three error patterns of the paper's Table 2 reappear: class noise rises almost linearly to random guessing, noise in every attribute levels off, and noise in one attribute saturates. Without the χ² test the tree grows past 1 000 nodes and errs more.
* **Unknown values.** Quinlan's proportional rule beats both "unknown as a value" and mode imputation when values are hidden at random (car at 30 %: 24.6 % error vs 29.0 % and 29.9 %).

## What is in this repository

| Path | Contents |
|---|---|
| `ID3_Project.ipynb` | The demo notebook: theory, every experiment, figures and discussion. Runs on Colab or a laptop. |
| `id3.py` | ID3 from scratch with NumPy, pandas and SciPy: entropy, gain, gain ratio, χ² test, unknown values, windowing, binary subset tests, continuous thresholds, noise corruption. |
| `experiments.py` | Produces every table in `results/` and every figure in `figures/`. |
| `simulate.py` | Generates the simulated data from a known decision tree (used only for sanity checks). |
| `tests/test_id3.py` | 15 unit tests, each checking a number or claim from the paper. |
| `data/real/` | The paper's Table 1 and 7 real UCI datasets (sources below). |
| `data/simulated/` | `sim_tree_2000.csv` (known tree, seed 2026) and `sim_with_id_600.csv` (adds a 150-valued irrelevant attribute, seed 7). |
| `results/`, `figures/` | Outputs of `experiments.py`. |
| `ID3_Slides.pptx`, `ID3_Slides.pdf` | The 8-slide presentation. |
| `ID3_Project_Submission.pdf` | The consolidated PDF: slides, then data details, all outputs and the code listing. |

## How to run

**Google Colab** (the notebook's first cell clones the repo automatically, so you can also just open the notebook in Colab and choose *Run all*)
```python
!git clone https://github.com/Vinayak069/id3-decision-trees
%cd id3-decision-trees
```
Then open `ID3_Project.ipynb` and choose *Runtime → Run all*. Alternatively, upload the folder to `My Drive/id3-project`; the first cell mounts Drive and moves into it. For a fast live demo set `QUICK = True` in the first cell (5 repetitions instead of 20).

**Laptop**
```bash
pip install -r requirements.txt
python -m pytest -q tests     # 15 tests, about 1 second
python simulate.py            # regenerates data/simulated (same seeds, same files)
python experiments.py         # all results/ and figures/ (add --quick for 5 repetitions)
jupyter notebook ID3_Project.ipynb
```
A full run takes about 8 minutes on a 4-core laptop. scikit-learn is used only for the CART baseline and to load the Wisconsin data.

## Using the tree directly
```python
import pandas as pd, id3
d = pd.read_csv("data/real/weather_quinlan1986.csv", dtype=str)
X, y = d[["outlook", "temperature", "humidity", "windy"]], d["class"]
tree = id3.ID3().fit(X, y)
print(tree.export_text())
tree.predict(pd.DataFrame([{"outlook": "?", "temperature": "hot", "humidity": "high", "windy": "false"}]))
```
Options of `id3.ID3`:
* `criterion="gain" | "gain_ratio"`
* `chi2_alpha=0.01` for the paper's 99 % test (default `None`: no test)
* `max_depth`, `min_samples_split`
* `split="multiway" | "subset"` (Section 7 binary tests A ∈ S)
* `continuous=[...]` for numeric columns, split at midpoint thresholds
* `empty_branch="majority" | "null"`

Unknown values are written as `?`. `id3.fit_windowed` implements windowing, and `id3.corrupt` / `id3.blank_out` inject noise and unknown values.

## Data sources

The UCI site was unreachable from our environment, so the files were taken from public GitHub copies of the UCI originals.

| Dataset | Objects | Original source | Copy used |
|---|---|---|---|
| Saturday mornings | 14 | Table 1 of the paper | typed in |
| Chess KR vs KP | 3 196 | UCI, Shapiro (1987), https://doi.org/10.24432/C5DK5C | mikeizbicki/datasets |
| Mushroom | 8 124 | UCI, Schlimmer (1981), https://doi.org/10.24432/C5959T | stedy/Machine-Learning-with-R-datasets |
| Congressional votes 1984 | 435 | UCI, Schlimmer (1987), https://doi.org/10.24432/C5C01P | mhahsler/arules |
| Car Evaluation | 1 728 | UCI, Bohanec & Rajkovic (1988), https://doi.org/10.24432/C5JP48 | scikit-learn-contrib/category_encoders |
| Tic-tac-toe endgame | 958 | UCI, Aha (1991), https://doi.org/10.24432/C5688J | LucasBoTang/Optimal_Classification_Trees |
| Breast cancer (Ljubljana) | 286 | UCI, Zwitter & Soklic (1988), https://doi.org/10.24432/C51P4M | LucasBoTang/Optimal_Classification_Trees |
| Breast cancer (Wisconsin) | 569 | UCI, Wolberg et al. (1995), https://doi.org/10.24432/C5DW2B | scikit-learn `load_breast_cancer` |

Mushroom, Votes and Ljubljana contain genuinely missing values (`?`). The simulated data is described in the docstring of `simulate.py`.

## Departures from the paper (labelled)
* Entropy, gain and the χ² statistic are written for K ≥ 2 classes. They reduce to the paper's formulas when K = 2.
* `max_depth` and `min_samples_split` are added for the depth-control experiment.
* The paper's own chess datasets (551 objects etc.) are not public. We use the public KR vs KP endgame and six other UCI datasets.
* For noise and unknown values we use a 70/30 train/test split with independent corruption of each part. The paper corrupted the whole set twice.
* Methods are compared on the same splits with the Nadeau–Bengio corrected resampled t-test, which the paper does not do.
