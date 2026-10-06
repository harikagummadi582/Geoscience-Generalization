# Geoscience-Generalization

CS 7980 Research Capstone, F26-3 "Tabular machine learning and data generators"
(PI: Dr. Tehmina Amjad). Contributors: Harika Gummadi, Samruddhi Kaledeshmukh, Aayush Katoch.

## Idea
Instead of predicting pore pressure per depth sample or over a whole well, represent each well as a
sequence of local depth segments (like temporal segmentation in video), learn within each segment,
and merge overlapping segment predictions into a full profile. Later: replace fixed windows with
data-driven geological segmentation (e.g. change-point detection) and test whether locally
homogeneous intervals improve accuracy, cross-well generalization and interpretability.

## Current status (prototype)
`src/lowo_window.py` runs leave-one-well-out (LOWO) XGBoost on the four Murree-formation wells:
- pointwise baseline vs fixed windows (10/25/50/100 m, 50% overlap, predictions averaged where windows overlap)
- full feature set (incl. HP, OB, DT_NCT, TVD) vs raw logs only
- Output: `results/lowo_results.csv`, `results/lowo_summary.csv`

Preliminary: 4 folds x 3 XGBoost seeds. Short windows (10 m) look better than pointwise on the full feature set (mean R² +0.10 vs -0.30), but 25-100 m do not improve consistently and gains vary by well. See `results/lowo_summary_3seeds.csv`.

## Layout
- `src/` code, `results/` metrics (small CSVs only), `data/` local data (git-ignored)

## Branch workflow
Work on `harika/*` branches, open PRs into `main`.
