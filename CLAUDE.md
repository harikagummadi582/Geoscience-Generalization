# Project context for Claude Code

CS 7980 Research Capstone (Northeastern), project F26-3 "Tabular machine learning and data generators".
PI: Dr. Tehmina Amjad. Team: Harika Gummadi (owner of this repo), Samruddhi Kaledeshmukh, Aayush Katoch.
Topic: pore-pressure prediction (PPP, psi) from well logs, "Generalization" sub-project.

## Research idea (PI's direction)
Instead of treating a well as one continuous prediction domain, or each depth sample independently,
represent each well as a sequence of local depth segments (like temporal segmentation in video).
Models learn pore-pressure relationships inside local windows; predictions from consecutive or
overlapping windows are merged into a full profile. Later: replace fixed windows with data-driven
geological segmentation (e.g. change-point detection on GR/DT/RHOB) and test whether locally
homogeneous intervals improve accuracy, cross-well generalization and interpretability.

## Data (NOT in this repo, never commit it)
21 Potwar Basin wells, one CSV each, in the course repo folder `GeoScience-main/data` (see its README).
- PPP is a *predicted* pore-pressure curve, not measured pressure: report "agreement with PPP".
- Missing values are `-999` (most files) or `-999.25` (MISSA KESWAL-01/02/03, QAZIAN -1X).
- Depth/TVD in meters. PINDORI-1/2 are sampled ~4x finer (0.0381 m) than most wells.
- Split by well, never by row. Fixed split: test = RAJIAN-03A, PINDORI-2, TURKWAL DEEP X 2,
  Balkassar POL 01; val = MINWAL-X-1, MINWAL-2, MISSA KESWAL-02, Balkassar OXY 01; train = other 13.
- Murree-formation depth ranges are applied to MISSA KESWAL-01/02/03 and QAZIAN -1X (as in the HME pipeline).
- Leakage columns excluded (MW, PP, FP, VES, DPHI, NPHI, ...). Check HME `preprocessing_pipeline.py` before adding any.

## Code
`src/lowo_window.py <data_dir> <out_dir> --mode {profile,fixed,lowo} [--features full,raw] [--seed N]`
- Cleans like the HME pipeline, decimates every well to ~0.1524 m, builds windows in METERS
  (10/25/50/100 m, 50% overlap, never spanning depth gaps), adds window-context features
  (mean/std/slope of GR, DT, SPHI, RHOB, RES + GR/DT anomaly), XGBoost, merges overlapping
  window predictions by averaging per sample.
- Feature sets: `full` (incl. HP, OB, DT_NCT, TVD, Eaton ratio, gradients) vs `raw` (logs only).
- Results: `results/` (4-well preview) and `results/21wells/` (21 wells, 3 seeds).

## Results so far (21 wells, XGBoost, mean over wells; 3 seeds)
Leave-one-well-out, full features:
| | R2 | RMSE (psi) |
|---|---|---|
| pointwise | 0.415 | 590 |
| window 10 m | 0.643 | 452 |
| window 25 m | 0.654 | 462 |
| window 50 m | 0.599 | 495 |
| window 100 m | 0.541 | 534 |
Windows beat pointwise on 21/21 (10 m), 20/21 (25, 50 m), 18/21 (100 m) wells.
Fixed split, test wells (full): pointwise R2 0.455 / RMSE 627; 10 m 0.461 / 528; 25 m 0.401 / 578;
50 m 0.503 / 590; 100 m 0.351 / 653. Raw-logs-only is negative R2 everywhere (inputs HP/OB/DT_NCT/TVD matter).

## Open issues / next tasks
1. **Smoothing control (not yet run):** averaging overlapping window predictions smooths output.
   Compare against pointwise predictions smoothed with a centred moving average of the same length,
   to show the gain is not just smoothing.
2. Add tests (`tests/`): windows never span a depth gap; no well in both train and test; merged
   predictions cover every sample once.
3. Data-driven segmentation (change-point detection) vs fixed windows.
4. Weekly Report 2 and Literature Review (Canvas, due Tue Oct 6 11:59 PM); Report 3 due Oct 12.

## Working agreements (see skills.md)
- Plan first: 3 lines (goal, inputs, how to check) before each task; one small change at a time.
- Show diffs; I (Harika) review `git diff` and write/approve commit messages. Don't push without being asked.
- Never commit data, `.pkl`, notebooks or large files. Work on `harika/*` branches, PR into `main`.
- Explain simply and concisely; say when a result is uncertain.
