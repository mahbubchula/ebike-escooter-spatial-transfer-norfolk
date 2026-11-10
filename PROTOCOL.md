# Analysis protocol

Fixed before modeling (November 9, 2026). Every later decision was recorded here before the test set was opened, and the results of each decision are dated.

## Question
In Norfolk's dockless Lime fleet, which time, weather, origin-destination and neighborhood factors distinguish
e-bike trips from e-scooter trips, are these relationships transferable across space (to neighborhoods the model
has not seen) and to a later period, and how do they vary across the city?

- RQ1 Determinants: what separates e-bike from e-scooter trips, conditional on how many e-bikes the operator deployed?
- RQ2 Spatial transferability (main contribution): does a model trained on some neighborhoods predict others, and
  where does it fail? Combined with a later-period test.
- RQ3 Spatial heterogeneity: how do explanations vary by tract; are there coherent "e-bike zones"?

## Unit, target, sample
- Unit: one trip. Target `e_bike` = 1 (Lime e-bike), 0 (Lime e-scooter).
- Sample: trips from 1 June 2021 (e-bikes introduced) to 31 October 2026 with known origin and destination census
  tract (80 tracts). November 2026 is partial and excluded. Cleaning rules as in `src/prep_03_clean_and_features.py`.
- E-bikes are 3-13% of trips depending on the quarter: the class is rare and its share is driven by fleet deployment.

## Leakage controls
1. Post-choice variables are never features: recorded trip distance (`distance_mi`, `dist_km`: distance actually
   ridden), duration, end hour.
2. Trip length enters only as the straight-line distance between origin and destination tract centroids (known
   before the trip, coarse) and a same-tract indicator.
3. Fleet supply is not observed. It is proxied only by strictly lagged quantities computed from trips on days
   t-7 ... t-1 (citywide e-bike share and trip count). Same-day or later trips are never used.
   An extension adds the origin tract's e-bike share over days t-28 ... t-1 ("local history"); it is excluded from the
   main model because a model for an unseen area cannot rely on that area's own history.
4. No outcome-based encoding of tracts. Tract identity enters only through ACS 2023 / LODES 2022 attributes
   (released before the study period ends; time-invariant; this is stated as a limitation for 2021-2022 trips).
5. Calendar `month` and `year` are not features (fleet and season changes are handled by weather and the supply proxy).
6. Feature filtering (|Spearman| >= 0.7, pre-declared priority) is computed on the inner training set only.
7. Imputation/scaling inside pipelines fitted on training data. Class weighting (if selected) only inside training.
8. Temporal order: no shuffling across time. Test set used once, protected by a SHA-256 model lock.
9. Spatial blocks are defined from tract centroids only (k-means on projected coordinates, seed 42), never from outcomes.

## Data split (temporal)
| Set | Period | Use |
|---|---|---|
| Inner train | 2021-06-01 to 2024-12-24 | fit candidate models |
| Embargo | 2024-12-25 to 2024-12-31 | discarded |
| Validation | 2025-01-01 to 2025-12-31 | early stopping, tuning, class-weight choice, threshold, spatial-CV evaluation |
| Final train | 2021-06-01 to 2025-12-31 | refit selected configuration |
| Test | 2026-01-01 to 2026-10-31 | evaluated once |

## Spatial transferability design (RQ2)
- Origin tracts are grouped into K = 6 contiguous spatial blocks (k-means on projected tract centroids).
- Pre-test (validation): for each block b, a "seen-space" model (trained on inner-train trips from all blocks) and an
  "unseen-space" model (trained on inner-train trips whose origin is NOT in b) are evaluated on validation trips
  starting in b. Transfer gap = metric(seen) - metric(unseen). Same hyperparameters for both.
- Post-test (reported, never used for choices): the same comparison with final-train -> test.

## Models
Dummy; logistic regression with splines; logistic regression with origin-tract fixed effects (spatial benchmark);
LightGBM; XGBoost; CatBoost. Best boosting model tuned with Optuna on validation (objective: log loss), with and
without class weighting.

## Metrics
PR-AUC (primary, rare class), ROC-AUC, log loss, Brier score, calibration intercept and slope, balanced accuracy,
F1 and MCC at a validation-chosen threshold. Day-block bootstrap 95% CIs. By month, by spatial block, by tract.

## Explanation plan (RQ1, RQ3)
TreeSHAP (global, interactions), ALE; tract-level mean SHAP profiles mapped; global Moran's I and local (LISA)
clusters of tract SHAP values and of tract residuals (queen contiguity); k-means clustering of tract SHAP profiles
into "zones". SHAP describes model behavior, not causal effects.

## Candidate results and decisions (validation only, 2026-11-09, test not opened)
- Class weighting (`is_unbalance`) destroyed calibration (validation calibration intercept +8.9, log loss 0.181 vs
  0.156) and is rejected; all models are trained unweighted.
- Untuned boosting stopped early (80-200 trees) and did not beat the spline logistic regression (validation PR-AUC
  0.078 vs 0.083; log loss 0.1558 vs 0.1529), consistent with a strong prior shift (7.7% e-bike in inner train vs 3.8%
  in validation).
- Next: tune LightGBM (Optuna, 40 trials, validation log loss), with an optional monotone-increasing constraint on the
  lagged citywide e-bike share (a fleet-supply prior). Selection rule (fixed now): the model with the lower validation
  log loss is primary; if the difference is < 0.001, the simpler model (spline logistic regression) is primary. The
  other model is reported as a comparison. Decision threshold: maximizes F1 on validation.

## Selection result (validation only, 2026-11-10, test not opened)
- Tuned LightGBM (40 Optuna trials; num_leaves 61, min_data_in_leaf 4774, monotone-increasing constraint on the lagged
  citywide e-bike share selected; 219 trees): validation PR-AUC 0.0825, AUC 0.698, log loss 0.15219.
- Spline logistic regression: PR-AUC 0.0827, AUC 0.692, log loss 0.15292.
- Log-loss difference -0.00073 is smaller than the pre-declared 0.001 margin -> PRIMARY MODEL = spline logistic
  regression; LightGBM is the comparison model. Threshold (max F1 on validation) = 0.082.
- Origin-only sensitivity models use the main model's settings without separate tuning (fitted before the test lock).
- Explanation sample (`step_07_xai_spatial.py`): 2026 test trips stratified by origin tract (<= 1,500 per tract) for stable tract
  profiles; global importance uses design weights (tract trips / sampled trips) so it represents all 2026 trips.
- Spatial transfer note: the lagged citywide supply proxy includes trips from the held-out block; this is information
  an operator has for any area, so it is kept. Origin-tract history (main_plus_history) is never used for unseen areas.
- Spatial transfer control (added before any transfer result was seen): a size-matched "seen" model trained on a random
  subsample of all training trips with the same size as the unseen-area training set. Transfer gap reported as
  matched - unseen (spatial novelty), with seen - unseen (novelty + data loss) as a secondary quantity.
- main_plus_history selection (validation): LightGBM PR-AUC 0.1017, AUC 0.719, log loss 0.14861; spline LR 0.1000,
  0.715, 0.14947; difference -0.00086 < 0.001 margin -> spline LR primary; threshold 0.090. Each feature set is
  evaluated at its own validation threshold (origin_only uses the main threshold, 0.082).
- Spatial transfer (validation) result: matched ~ seen in every block (data loss is not the issue); unseen-area penalty
  (matched - unseen) positive for the spline LR in all six blocks (AUC 0.006-0.110), smaller for LightGBM and absent in
  blocks 2 and 5; calibration intercept in unseen central blocks -3.3 (LR) and -1.7 (LightGBM). Primary model is not
  changed by this result (rule fixed in advance); both models are reported.
- Test (once, 2026-11-10): primary spline LR PR-AUC 0.093 [0.088, 0.099], AUC 0.653; LightGBM 0.098 [0.092, 0.103],
  AUC 0.663 (paired PR-AUC difference 0.004 to 0.006). Spatial transfer on test reproduces validation. No model changed.
- Explanations (`step_07_xai_spatial.py`) are computed for both models; tract zones are formed from each model's SHAP profiles with the
  same k and compared with the adjusted Rand index.
- Robustness (`step_09_robust_blocks.py`, validation only): spatial-transfer protocol repeated with K = 4 and K = 8 blocks.
  Result: trip-weighted AUC penalty (matched - unseen) LR / LightGBM = 0.138 / 0.054 (K=4), 0.082 / 0.043 (K=6),
  0.073 / 0.041 (K=8); the spline LR pays roughly twice the LightGBM penalty at every K; penalties concentrate in the
  large central blocks.
