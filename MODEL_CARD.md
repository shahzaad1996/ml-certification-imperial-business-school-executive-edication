# Model Card: BBO Multi-Surrogate Optimisation Pipeline

> Framework: Mitchell et al. (2019) "Model Cards for Model Reporting"

---

## 1. Overview

| Field | Detail |
|---|---|
| **Name** | BBO Multi-Surrogate Optimisation Pipeline |
| **Type** | Sequential Black-Box Optimisation (Bayesian Optimisation + ensemble surrogates) |
| **Version** | v9 (final, after nine query rounds) |
| **Author** | Capstone student, Imperial Business School Executive Education |
| **Language / stack** | Python 3.14 · scikit-learn · NumPy · Matplotlib · Seaborn |
| **Repository** | See README for GitHub link |

---

## 2. Intended Use

**Suitable tasks:**
- Sequential optimisation of expensive black-box functions where the number of evaluations is strictly limited (budget ≤ ~50).
- Functions with continuous inputs normalised to `[0, 1]^d`, for any `d` from 2 to at least 8.
- Settings where no gradient information is available and the function may be noisy, multi-modal, or signal-sparse.

**Use cases to avoid:**
- Functions with discrete or categorical inputs (the GP RBF kernel and UCB acquisition assume continuous, smooth spaces).
- Problems where `d > 10` or `n_observations < 10` — the GP will be under-constrained and the MLP bootstrap ensemble will have very high variance.
- Real-time or latency-sensitive applications: the SVR ensemble (25 models) and MLP ensemble (50 models) have non-trivial training times on large candidate grids.
- Any safety-critical decisions without independent validation, regardless of surrogate performance metrics.

---

## 3. Strategy Details Across Nine Rounds

### Evolution of approach

| Round | Surrogate(s) | Acquisition | Key change |
|---|---|---|---|
| 1 | GP (RBF + WhiteKernel) | UCB, β = 30 (2D grid) | Baseline; broad exploration |
| 2–3 | GP | UCB, high β | Refined kernel restarts (10), `normalize_y=True` |
| 4 | GP + SVR bootstrap (25 models) | UCB per surrogate | SVR ensemble added for empirical uncertainty |
| 5–6 | GP + SVR + MLP bootstrap (50 models) | UCB + Expected Improvement | MLP ensemble added; TensorFlow replaced by `sklearn.neural_network.MLPRegressor` |
| 7–8 | All three surrogates | Mixed UCB; classification layer (LR + SVC + `MLPClassifier`) for F1 | Classification acquisition added for signal-sparse functions |
| 9 | All three, consensus vote | UCB with per-function β/κ schedule | β reduced on well-characterised functions; κ = 10 maintained on high-dim |

### Surrogate specifications (final)

**Gaussian Process (primary — low-dim):**
- Kernel: `1.0 × RBF(length_scale=0.2, bounds=(1e-2, 1.0)) + WhiteKernel(noise_level=1e-6, bounds=(1e-10, 1e-2))`
- `normalize_y=True`, `n_restarts_optimizer=10`, `random_state=42`
- Used for: F1–F4 (2D–4D), F8 (8D) as primary

**SVR Bootstrap Ensemble:**
- 25 models, `sklearn.svm.SVR(kernel='rbf', C=10.0, epsilon=1e-3, gamma='scale')`
- Each model trained on a bootstrap resample of the full dataset
- Uncertainty: standard deviation across ensemble predictions

**MLP Bootstrap Ensemble:**
- 50 models, `sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(64,64), activation='relu', max_iter=500)`
- Each model trained on a bootstrap resample
- Uncertainty: standard deviation across ensemble predictions
- Replaces MC-Dropout / Keras (incompatible with Python 3.14)

**Classification layer (F1 only):**
- Binary labels: `y ≥ 75th percentile` → class 1
- Models: `LogisticRegression(solver='lbfgs')`, `SVC(kernel='rbf', probability=True)`, `MLPClassifier(hidden_layer_sizes=(32,16))`
- Acquisition: `score = p_class1 + β × normalised_predictive_entropy`

### Candidate generation

| Function dim | Candidate method | N candidates |
|---|---|---|
| 2D (F1–F2) | Regular 100×100 grid | 10,000 |
| 3D–5D (F3–F6) | Random uniform | 5,000 |
| 6D–8D (F7–F8) | Random uniform | 50,000 |

---

## 4. Performance

### Best observed outputs per function (after 9 rounds)

| Function | Scenario | Best output observed | Input at best |
|---|---|---|---|
| F1 | 2D radiation detection | −0.00765 | `[0.6465, 0.6768]` (wk13) |
| F2 | 2D signal field | 0.611 (initial) → improved | see notebook |
| F3 | 3D unknown | −0.035 (initial best retained) | see notebook |
| F4 | 4D unknown | −4.026 (initial best retained) | see notebook |
| F5 | 4D chemical yield | **6117.3** | `[0.886, 0.998, 0.960, 0.993]` (wk15) |
| F6 | 5D unknown | −0.714 (initial best retained) | see notebook |
| F7 | 6D unknown | 1.365 (initial best retained) | see notebook |
| F8 | 8D hyperparameter tuning | **9.895** | `[0.014, 0.203, 0.064, 0.132, 0.951, 0.485, 0.038, 0.914]` (wk18) |

### Metrics used
- **Primary**: best-observed output value `y*` after each round (best-so-far curve).
- **Acquisition quality**: UCB score at the selected candidate vs the grid maximum.
- **Uncertainty proxy**: ensemble standard deviation `σ` at `x_next`; higher σ = more exploratory query.
- No held-out test set is used (the black-box is the ground truth; no labels exist for unqueried points).

### Observed patterns
- **F5**: Steady improvement across rounds driven by GP-UCB; best output grew from ~1089 (initial) to 6117 (wk15), confirming the unimodal peak in the high-end region of all four inputs.
- **F8**: GP-UCB with κ = 10 over 50,000 candidates produced consistent outputs in the range 7.3–9.9, confirming the surrogate is identifying productive 8D subspaces.
- **F1**: Diminishing returns after wk13; outputs collapsed to ~0 by wk19–20, indicating the non-zero support is very localised. Classification acquisition now used to re-localise the source.

---

## 5. Assumptions and Limitations

### Assumptions
1. **Smoothness**: All surrogates assume the objective function is locally smooth. If the function is discontinuous or has sharp ridges finer than the GP length-scale, the GP will under-fit.
2. **Stationarity**: The RBF kernel assumes the smoothness is constant across the domain. Non-stationary functions (e.g., with a very sharp peak in one region and a flat plateau elsewhere) will be mis-specified.
3. **Independence across functions**: Each of the eight functions is optimised independently. No transfer learning or multi-task GP is used.
4. **Query feedback is exact**: Outputs are treated as noise-free or low-noise. The WhiteKernel accounts for small noise but the pipeline is not designed for high-noise oracles.
5. **Uniform candidate grid**: For high-dim functions (F7–F8), candidates are uniform random draws. If the true optimum is in a very small region, it may not be sampled as a candidate and will be missed.

### Limitations and failure modes
- **Small dataset size**: With 10–50 observations, bootstrap ensemble variance estimates are noisy. The GP is the most reliable uncertainty model at this scale.
- **No batch acquisition**: One point is submitted per round per function. Batch BO (selecting multiple non-redundant points simultaneously) would be more efficient.
- **Bootstrap ≠ Bayesian uncertainty**: The SVR and MLP ensemble standard deviations are frequentist approximations, not calibrated posterior estimates. They may over- or under-estimate true uncertainty.
- **Python 3.14 incompatibility**: TensorFlow / Keras could not be used. The MLP bootstrap ensemble is a functional replacement but does not support true MC-Dropout inference.
- **No Optuna / hyperparameter tuning**: Surrogate hyperparameters (C, hidden layer size, β/κ) were set manually. Automated tuning would likely improve acquisition quality.

---

## 6. Ethical Considerations

**Transparency and reproducibility:**  
All query inputs, outputs, surrogate specifications, acquisition parameters and random seeds (`random_state=42` throughout) are recorded in the Jupyter notebooks and this model card. Any reader can re-run the notebooks from the provided `.npy` files and reproduce every query decision.

**Fairness across functions:**  
The same surrogate pipeline is applied to all eight functions. No function receives preferential treatment in terms of query budget allocation — each receives exactly one query per round. This is a deliberate design choice to ensure fair comparison of surrogate performance across different problem structures.

**Real-world adaptation:**  
The function scenarios (radiation detection, chemical yield, hyperparameter tuning) are pedagogical proxies. Before adapting this pipeline to any real-world system, the following would be required:
- Domain-specific noise modelling (replacing the generic WhiteKernel).
- Safety constraints (e.g., chemical process inputs may have toxicity or stability bounds not captured by `[0, 1]` normalisation).
- Validation against held-out real evaluations before deployment.

**Limitations of model card:**  
This card describes the optimisation strategy, not a trained predictive model with fixed weights. The surrogates are retrained from scratch at each round, so there is no persistent "model" to deploy — the card therefore documents the *process* rather than a specific artefact. This is appropriate given the sequential, iterative nature of BBO.
