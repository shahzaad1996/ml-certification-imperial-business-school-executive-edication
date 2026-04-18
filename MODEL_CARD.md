# Model Card: BBO Multi-Surrogate Optimisation Pipeline

> Framework: Mitchell et al. (2019) "Model Cards for Model Reporting"

---

## 1. Overview

| Field | Detail |
|---|---|
| **Name** | BBO Multi-Surrogate Optimisation Pipeline |
| **Type** | Sequential Black-Box Optimisation (Bayesian Optimisation + ensemble surrogates) |
| **Version** | v10 (final, after ten query rounds — weeks 13–21) |
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
- Real-time or latency-sensitive applications: the SVR ensemble (25 models) and MLP ensemble (50 models) have non-trivial training times on large candidate grids (50,000 candidates for 8D).
- Any safety-critical decisions without independent validation, regardless of surrogate performance metrics.

---

## 3. Strategy Details Across Ten Rounds

### Evolution of approach

| Round | Surrogate(s) | Acquisition | Key change |
|---|---|---|---|
| 1 | GP (RBF + WhiteKernel) | UCB, β = 30 (2D grid) | Baseline; broad exploration |
| 2–3 | GP | UCB, high β | Refined kernel restarts (10), `normalize_y=True` |
| 4 | GP + SVR bootstrap (25 models) | UCB per surrogate | SVR ensemble added for empirical uncertainty |
| 5–6 | GP + SVR + MLP bootstrap (50 models) | UCB + Expected Improvement | MLP ensemble added; TensorFlow replaced by `sklearn.neural_network.MLPRegressor` |
| 7–8 | All three surrogates | Mixed UCB; classification layer (LR + SVC + `MLPClassifier`) for F1 | Classification acquisition added for signal-sparse functions |
| 9 | All three, consensus vote | UCB with per-function β/κ schedule | β reduced on well-characterised functions; κ = 10 maintained on high-dim |
| 10 | All three surrogates | Per-function κ/β (see table below) | Focus radius halved (0.01–0.015); step sizes prioritised over direction; final exploitation pass |

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

| Function | Dim | Candidate method | N candidates | κ / β |
|---|---|---|---|---|
| F1 | 2D | Regular 100×100 grid | 10,000 | β = 6 (GP/SVR/MLP); entropy (classification) |
| F2 | 2D | Regular 100×100 grid | 10,000 | κ = 3 |
| F3 | 3D | Random uniform | 5,000 | κ = 30 (high exploration) |
| F4 | 4D | Random uniform | 5,000 | κ = 3 |
| F5 | 4D | Random uniform | 5,000 | κ = 4 |
| F6 | 5D | Random uniform | 20,000 | κ = 4 |
| F7 | 6D | Random uniform | 20,000 | κ = 3 |
| F8 | 8D | Random uniform | 50,000 | κ = 10 |

---

## 4. Performance

### Best observed outputs per function (after 10 rounds, weeks 13–21)

| Function | Scenario | Best output | Best input | Best week |
|---|---|---|---|---|
| F1 | 2D contamination detection | −0.00765 | `[0.646, 0.677]` | wk13 |
| F2 | 2D noisy log-likelihood | **0.697** | `[0.859, 0.343]` | wk20 |
| F3 | 3D drug compound minimisation | **−0.056** | `[0.448, 0.218, 0.560]` | wk13 |
| F4 | 4D warehouse tuning | **0.303** | `[0.404, 0.434, 0.436, 0.384]` | wk20 |
| F5 | 4D chemical yield | **6117.3** | `[0.886, 0.998, 0.960, 0.993]` | wk15 |
| F6 | 5D cake recipe | **−0.337** | `[0.399, 0.372, 0.622, 0.993, 0.189]` | wk17 |
| F7 | 6D ML hyperparameters | **2.791** | `[0.022, 0.239, 0.465, 0.283, 0.347, 0.636]` | wk20 |
| F8 | 8D black-box optimisation | **9.895** | `[0.014, 0.203, 0.064, 0.132, 0.951, 0.485, 0.038, 0.914]` | wk18 |

### Metrics used
- **Primary**: best-observed output value `y*` after each round (best-so-far curve).
- **Acquisition quality**: UCB score at the selected candidate vs the grid maximum.
- **Uncertainty proxy**: ensemble standard deviation `σ` at `x_next`; higher σ = more exploratory query.
- No held-out test set is used (the black-box is the ground truth; no labels exist for unqueried points).

### Observed patterns
- **F1 (2D, contamination)**: Signal is extremely localised. Outputs from wk16 onward are indistinguishable from zero (range 1e-53 to 3.38e-96). The strongest signal (−0.00765 at wk13) was never surpassed despite multiple surrogate approaches (GP, SVR ensemble, MLP ensemble, classification with predictive entropy). The non-zero support of this function is very narrow.
- **F2 (2D, log-likelihood)**: Progressive improvement across rounds; wk20 achieved 0.697 at `[0.859, 0.343]`. Next GP-UCB suggestion is very close to best (`[0.869, 0.303]`), indicating convergence. Outputs are variable but the algorithm identified a productive region.
- **F3 (3D, drug compounds)**: Best result (−0.056) found in the first student query (wk13). Subsequent exploration worsened results, with wk20 producing −0.248. Lower Compound B values correlate with less-negative outputs.
- **F4 (4D, warehouse)**: Massive volatility (−33.24 to +0.303). Sharp landscape cliffs make the GP overconfident. Safe region identified around [0.35–0.45] across all four parameters; boundary exploration (wk14–16) caused severe regressions.
- **F5 (4D, chemical yield)**: Unimodal peak confirmed in the upper corner of the hypercube. Best output grew from ~1089 (initial) to 6117 (wk15), confirming the peak at [0.88–1.0]⁴. Exploratory probes to low values (wk18: 171; wk20: 103) were catastrophic but informative.
- **F6 (5D, cake recipe)**: Consistently negative outputs with no clear improvement trajectory. Best at −0.337 (wk17) requires high butter (~0.99), moderate eggs (~0.62), low milk (~0.19). Extreme ingredient values (wk21: flour=0.89, butter=0.003) produce the worst results (−2.75).
- **F7 (6D, ML hyperparameters)**: Strong improvement from wk19 onward (outputs 2.3–2.8 vs 0.27–1.8 in wk13–18). Best at wk20 (2.791) with very low HP1 (~0.02) and moderate other parameters. Algorithm converging to a productive 6D subspace.
- **F8 (8D, black-box)**: GP-UCB with κ = 10 over 50,000 candidates produced the overall peak (9.895) at wk18. Parameters 5 (0.951) and 8 (0.914) appear to dominate the output. Post-peak exploration (wk19–21: 7.3–8.4) confirmed the wk18 region is competitive but could not surpass it.

---

## 5. Assumptions and Limitations

### Assumptions
1. **Smoothness**: All surrogates assume the objective function is locally smooth. If the function is discontinuous or has sharp ridges finer than the GP length-scale, the GP will under-fit.
2. **Stationarity**: The RBF kernel assumes the smoothness is constant across the domain. Non-stationary functions (e.g., with a very sharp peak in one region and a flat plateau elsewhere) will be mis-specified.
3. **Independence across functions**: Each of the eight functions is optimised independently. No transfer learning or multi-task GP is used.
4. **Query feedback is exact**: Outputs are treated as noise-free or low-noise. The WhiteKernel accounts for small noise but the pipeline is not designed for high-noise oracles.
5. **Uniform candidate grid**: For high-dim functions (F7–F8), candidates are uniform random draws. If the true optimum is in a very small region, it may not be sampled as a candidate and will be missed.

### Limitations and failure modes
- **Small dataset size**: With 19–49 observations, bootstrap ensemble variance estimates are noisy. The GP is the most reliable uncertainty model at this scale.
- **Isotropic RBF kernel**: The isotropic kernel assumes all input dimensions are equally important. For F8, parameters 5 and 8 appear to dominate the output; an ARD (Automatic Relevance Determination) kernel would learn per-dimension length scales and direct exploration more efficiently.
- **No batch acquisition**: One point is submitted per round per function. Batch BO (selecting multiple non-redundant points simultaneously) would be more efficient.
- **Bootstrap ≠ Bayesian uncertainty**: The SVR and MLP ensemble standard deviations are frequentist approximations, not calibrated posterior estimates. They may over- or under-estimate true uncertainty.
- **Python 3.14 incompatibility**: TensorFlow / Keras could not be used. The MLP bootstrap ensemble is a functional replacement but does not support true MC-Dropout inference.
- **No Optuna / hyperparameter tuning**: Surrogate hyperparameters (C, hidden layer size, β/κ) were set manually. Automated tuning would likely improve acquisition quality.
- **Overconfident GP on sharp landscapes**: F4's volatility (−33.24 to +0.303) demonstrates that the GP's smoothness assumption can be severely violated, leading to overconfident acquisition scores near landscape cliffs.

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
