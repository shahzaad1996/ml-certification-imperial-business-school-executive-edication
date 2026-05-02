# Model Card: BBO Multi-Surrogate Optimisation Pipeline

> Framework: Mitchell et al. (2019) "Model Cards for Model Reporting"

---

## 1. Overview

| Field | Detail |
|---|---|
| **Name** | BBO Multi-Surrogate Optimisation Pipeline |
| **Type** | Sequential Black-Box Optimisation (Bayesian Optimisation + ensemble surrogates) |
| **Version** | v13 (after twelve query rounds — weeks 13–24) |
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
| 10 | All three surrogates | Per-function κ/β (see table below) | Focus radius halved (0.01–0.015); step sizes prioritised over direction; exploitation pass |
| 11 | All three surrogates | Per-function κ/β (unchanged) | Clustering-informed refinement: queries target identified cluster centroids and extend trajectories; wk22 data appended |
| 12 | All three surrogates | Per-function κ/β (F4: κ=4, F5: κ=8; others unchanged) | Boundary probing and validation: systematic edge-of-region probes, reproducibility re-query (F7), increased κ on key functions; wk23 data appended |
| 13 | All three surrogates | Per-function κ/β (unchanged from round 12) | Exploitation refinement + extremes testing: targeted exploitation near F7 optimum (**new best 2.908**), extreme-parameter probes on F3/F5/F6/F8; wk24 data appended |

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
| F4 | 4D | Random uniform | 5,000 | κ = 4 |
| F5 | 4D | Random uniform | 5,000 | κ = 8 (high exploration) |
| F6 | 5D | Random uniform | 20,000 | κ = 4 |
| F7 | 6D | Random uniform | 20,000 | κ = 3 |
| F8 | 8D | Random uniform | 50,000 | κ = 10 |

---

## 4. Performance

### Best observed outputs per function (after 12 rounds, weeks 13–24)

| Function | Scenario | Best output | Best input | Best week | wk24 output |
|---|---|---|---|---|---|
| F1 | 2D contamination detection | −0.00765 | `[0.646, 0.677]` | wk13 | 0 |
| F2 | 2D noisy log-likelihood | **0.697** | `[0.859, 0.343]` | wk20 | 0.102 |
| F3 | 3D drug compound minimisation | **−0.056** | `[0.448, 0.218, 0.560]` | wk13 | −0.508 |
| F4 | 4D warehouse tuning | **0.303** | `[0.404, 0.434, 0.436, 0.384]` | wk20 | −0.163 |
| F5 | 4D chemical yield | **6117.3** | `[0.886, 0.998, 0.960, 0.993]` | wk15 | 2640 |
| F6 | 5D cake recipe | **−0.337** | `[0.399, 0.372, 0.622, 0.993, 0.189]` | wk17 | −2.526 |
| F7 | 6D ML hyperparameters | **2.908** | `[0.219, 0.221, 0.481, 0.355, 0.355, 0.597]` | wk24 | **2.908** |
| F8 | 8D black-box optimisation | **9.895** | `[0.014, 0.203, 0.064, 0.132, 0.951, 0.485, 0.038, 0.914]` | wk18 | 7.025 |

### Metrics used
- **Primary**: best-observed output value `y*` after each round (best-so-far curve).
- **Acquisition quality**: UCB score at the selected candidate vs the grid maximum.
- **Uncertainty proxy**: ensemble standard deviation `σ` at `x_next`; higher σ = more exploratory query.
- No held-out test set is used (the black-box is the ground truth; no labels exist for unqueried points).

### Observed patterns
- **F1 (2D, contamination)**: Signal is extremely localised. Outputs from wk16–wk22 were indistinguishable from zero; wk23 returned the first non-trivial signal since wk15 (−8.17e-5 at `[0.495, 0.404]`). The wk24 north-edge probe `[0.566, 1.0]` returned exactly zero, confirming the signal tail does not extend to the upper boundary. The non-zero support remains concentrated around `[0.65, 0.68]` with a possible low-amplitude tail toward the south-west.
- **F2 (2D, log-likelihood)**: Best remains 0.697 at wk20 `[0.859, 0.343]`. The wk24 low-x2 probe `[0.566, 0.192]` returned only 0.102, confirming the productive ridge requires both x1 > 0.85 and x2 ≈ 0.30–0.40. Queries outside this narrow band (wk23: high x1 → negative; wk24: low x1+x2 → weak) consistently underperform.
- **F3 (3D, drug compounds)**: Best (−0.056) from wk13. The wk24 extreme high-C probe `[0.398, 0.013, 0.995]` returned −0.508 — worst student result, 9× worse than best. Very high Compound C (>0.99) is strongly detrimental. Optimal C must stay in the 0.49–0.56 range; both extremes are harmful.
- **F4 (4D, warehouse)**: Massive volatility (−33.24 to +0.303). The wk24 query `[0.461, 0.409, 0.352, 0.423]` returned −0.163, confirming that HP1 > 0.45 crosses the landscape cliff. The viable positive zone is extremely narrow around HP1 ≈ 0.39–0.44.
- **F5 (4D, chemical yield)**: Unimodal peak at 6117 (wk15). The wk24 query with P3=0.097 returned only 2640, confirming that even one low parameter halves the yield. All four parameters must simultaneously exceed ~0.85.
- **F6 (5D, cake recipe)**: Best at −0.337 (wk17). The wk24 probe with near-zero sugar (0.044) and eggs (0.031) returned −2.526, revealing sugar and eggs as critical positive drivers alongside flour and butter. High milk (0.993) also strongly detrimental. Optimal recipe: flour 0.35–0.40, sugar >0.37, eggs >0.37, butter >0.99, milk <0.19.
- **F7 (6D, ML hyperparameters)**: **New best at wk24: 2.908** at `[0.219, 0.221, 0.481, 0.355, 0.355, 0.597]`, surpassing wk20's 2.791 by 4.2%. Higher HP1 (0.22 vs 0.02) and lower HP6 (0.60 vs 0.64) — the productive subspace is broader than initially characterised: `[0.02–0.22, 0.15–0.24, 0.33–0.48, 0.28–0.41, 0.21–0.36, 0.60–0.72]`.
- **F8 (8D, black-box)**: Peak at wk18 (9.895). The wk24 high-P2+P4+P6+P7 test returned only 7.025 — lowest in recent rounds. High P7 (0.976) appears detrimental; P5 and P8 remain the strongest drivers, P6 is secondary, and P7 is likely neutral or negative.

---

## 5. Assumptions and Limitations

### Assumptions
1. **Smoothness**: All surrogates assume the objective function is locally smooth. If the function is discontinuous or has sharp ridges finer than the GP length-scale, the GP will under-fit.
2. **Stationarity**: The RBF kernel assumes the smoothness is constant across the domain. Non-stationary functions (e.g., with a very sharp peak in one region and a flat plateau elsewhere) will be mis-specified.
3. **Independence across functions**: Each of the eight functions is optimised independently. No transfer learning or multi-task GP is used.
4. **Query feedback is exact**: Outputs are treated as noise-free or low-noise. The WhiteKernel accounts for small noise but the pipeline is not designed for high-noise oracles.
5. **Uniform candidate grid**: For high-dim functions (F7–F8), candidates are uniform random draws. If the true optimum is in a very small region, it may not be sampled as a candidate and will be missed.

### Limitations and failure modes
- **Small dataset size**: With 22–52 observations, bootstrap ensemble variance estimates are noisy. The GP is the most reliable uncertainty model at this scale.
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
