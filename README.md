# 🏀 Which NBA Superstar Delivers in the Clutch?

A data-analysis project that answers one hard question: **of the league's true #1 / #2
options, which superstar actually delivers when the pressure is highest?** It pulls five
seasons of NBA **playoff** data, restricts the field to each team's primary options (not role
players), engineers clutch-delivery metrics, and ranks players with a weighted **Composite
Clutch Score** — then proves the ranking is robust by re-scoring it three independent ways.

**Seasons covered:** 2021-22 → 2025-26 playoffs (five complete postseasons).

---

## 📈 Key results

The leaderboard is reported as an **ensemble** of three scoring methods, so it doesn't hinge
on any single set of weights. Top of the board:

| Rank | Player | Team | Ensemble | Notes |
|------|--------|------|---------:|-------|
| 1 | Shai Gilgeous-Alexander | OKC | 100.0 | Tops every method |
| 2 | Jalen Brunson | NYK | 95.9 | #1 on the raw composite (82.9); 2025-26 champion |
| 3 | Stephen Curry | GSW | 91.8 | |
| 4 | Jayson Tatum | BOS | 83.7 | |
| 5 | James Harden | CLE | 73.5 | |
| 5 | Nikola Jokić | DEN | 73.5 | |

The three scoring methods agree at **Spearman 0.88–0.94** — high agreement means the ranking
is a property of the data, not an artifact of arbitrary weighting choices.

---

## ❓ How "clutch" and "superstar" are defined

Precise definitions are what keep this project honest:

- **Clutch** — the NBA's standard definition: the **last 5 minutes** of a game with the score
  **within 5 points**.
- **Superstar / primary option** — one of a team's **top 2** players that season, ranked by a
  blend of **usage rate** and **points per game**. This deliberately filters out efficient
  low-volume role players — a catch-and-shoot specialist is not a "clutch superstar."
  Eligibility is computed *per season*, so a player only earns credit in the years he actually
  led a team.

---

## 🔬 Methodology

1. **Collect** — playoff clutch stats (Base / Advanced / Scoring), full-season context stats,
   and per-game logs from the NBA Stats API (`nba_api`). Every response is cached to
   `data/*.csv`, with a sleep inserted *only* on real network hits to respect rate limits.
2. **Filter** — keep each team's top-2 usage+scoring options per season; everyone else is
   dropped before scoring.
3. **Engineer features** — aggregate multi-season clutch rows into one career-clutch row per
   player using correctly-weighted averages (TS%/usage by clutch minutes, FT% by attempts,
   per-game stats by games). Derived metrics include **elimination uplift** (Game 6/7 Game
   Score vs. a non-elimination baseline) and **consistency** (from the coefficient of
   variation of per-game scoring).
4. **Score** — convert each metric to an outlier-robust **0–100 percentile** within the star
   pool, weight them (PPG 20%, TS% 20%, elimination 20%, usage 15%, +/- 10%, consistency 10%,
   win-rate 5% — team-driven stats deliberately down-weighted), then apply a **capped
   sample-size multiplier** so deep playoff runs can't win on raw game-count alone.
5. **Validate** — cross-check with multiple models (below) and an ensemble agreement test.

---

## 🤖 Models & techniques

| Model / technique | Library | Purpose |
|---|---|---|
| Weighted percentile composite | pandas / numpy | The headline Composite Clutch Score |
| **PCA** | scikit-learn | Finds dominant axes of variation (used to *understand*, not rank) |
| **K-Means** (silhouette-selected *k*) | scikit-learn | Surfaces clutch player **archetypes** |
| **Linear Regression** (5-fold CV R²) | scikit-learn | Which clutch skills track winning |
| **3-method ensemble** + Spearman | scipy / pandas | Robustness check on the ranking |

Supporting: `StandardScaler`, winsorization (5th/95th-pct clipping), weighted aggregation.

---

## 📊 Visualizations

The notebook produces nine charts, including:
- Top-20 clutch leaderboard
- Clutch efficiency (TS%) vs. impact (+/-), sized by usage and colored by composite
- Elimination-game specialists (Game 6/7 uplift)
- Most consistent playoff scorers
- Year-over-year clutch trend for the top 10
- PCA loadings, K-Means archetypes, regression coefficients, and a method-agreement heatmap

---

## 🚀 Getting started

```bash
# 1. (optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. open the notebook
jupyter notebook nba_clutch_analysis.ipynb
```

The `data/` directory ships with cached API responses, so the notebook runs **offline** out
of the box. To pull fresh data, delete the relevant `data/*.csv` files and re-run — the
notebook will re-fetch from the NBA API automatically.

### Regenerating the notebook

The notebook is generated from a single source-of-truth script, which keeps the JSON
well-formed and version-controllable:

```bash
python build_notebook.py          # regenerate nba_clutch_analysis.ipynb
jupyter nbconvert --to notebook --execute --inplace nba_clutch_analysis.ipynb
```

---

## 📁 Project structure

```
.
├── nba_clutch_analysis.ipynb   # the analysis (run this)
├── build_notebook.py           # source-of-truth generator for the notebook
├── data/                       # cached NBA API responses (CSV)
├── requirements.txt
└── README.md
```

---

## ⚠️ Limitations

Honest accounting of where the ranking can mislead — and how each is mitigated — lives in a
dedicated section of the notebook. Highlights: team-quality confounds in +/- and win-rate
(deliberately down-weighted), the binary top-2 eligibility cutoff, small clutch samples
(50-minute floor + capped multiplier + percentile normalization), and the use of full-game
Game Score as the elimination proxy.

---

## 🛠️ Tech stack

Python 3.10 · pandas · NumPy · scikit-learn · SciPy · matplotlib · seaborn · nba_api · Jupyter

## 📄 License

[MIT](LICENSE)
