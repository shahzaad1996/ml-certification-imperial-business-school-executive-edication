# Model Card: BBO Multi-Surrogate Optimisation Pipeline

> Framework: Mitchell et al. (2019) "Model Cards for Model Reporting"

---

## 1. Overview

| Field | Detail |
|---|---|
| **Name** | BBO Multi-Surrogate Optimisation Pipeline |
| **Type** | Sequential Black-Box Optimisation (Bayesian Optimisation + ensemble surrogates) |
| **Version** | v12 (after eleven query rounds — weeks 13–23) |
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

### Best observed outputs per function (after 11 rounds, weeks 13–23)

| Function | Scenario | Best output | Best input | Best week | wk23 output |
|---|---|---|---|---|---|
| F1 | 2D contamination detection | −0.00765 | `[0.646, 0.677]` | wk13 | −8.17e-5 |
| F2 | 2D noisy log-likelihood | **0.697** | `[0.859, 0.343]` | wk20 | −0.053 |
| F3 | 3D drug compound minimisation | **−0.056** | `[0.448, 0.218, 0.560]` | wk13 | −0.077 |
| F4 | 4D warehouse tuning | **0.303** | `[0.404, 0.434, 0.436, 0.384]` | wk20 | −0.077 |
| F5 | 4D chemical yield | **6117.3** | `[0.886, 0.998, 0.960, 0.993]` | wk15 | 163.0 |
| F6 | 5D cake recipe | **−0.337** | `[0.399, 0.372, 0.622, 0.993, 0.189]` | wk17 | −1.120 |
| F7 | 6D ML hyperparameters | **2.791** | `[0.022, 0.239, 0.465, 0.283, 0.347, 0.636]` | wk20 | 2.652 |
| F8 | 8D black-box optimisation | **9.895** | `[0.014, 0.203, 0.064, 0.132, 0.951, 0.485, 0.038, 0.914]` | wk18 | 9.315 |

### Metrics used
- **Primary**: best-observed output value `y*` after each round (best-so-far curve).
- **Acquisition quality**: UCB score at the selected candidate vs the grid maximum.
- **Uncertainty proxy**: ensemble standard deviation `σ` at `x_next`; higher σ = more exploratory query.
- No held-out test set is used (the black-box is the ground truth; no labels exist for unqueried points).

### Observed patterns
- **F1 (2D, contamination)**: Signal is extremely localised. Outputs from wk16–wk22 were indistinguishable from zero (range 1e-53 to exactly 0). The wk23 probe at `[0.495, 0.404]` returned −8.17e-5 — the first non-trivial signal since wk15 and the third-strongest ever detected. This suggests the non-zero support extends beyond the narrow `[0.65, 0.68]` zone, with a possible low-amplitude tail toward the south-west. The strongest signal (−0.00765 at wk13) was never surpassed.
- **F2 (2D, log-likelihood)**: Progressive improvement across rounds; wk20 achieved 0.697 at `[0.859, 0.343]`. The wk23 boundary probe at `[0.970, 0.374]` returned −0.053 — the first negative output observed — confirming the productive ridge drops off sharply beyond x1 ≈ 0.87. The optimal zone is concentrated around `[0.85–0.87, 0.30–0.40]`.
- **F3 (3D, drug compounds)**: Best result (−0.056) found in the first student query (wk13). The wk23 query `[0.977, 0.007, 0.491]` returned −0.077 (second-best student result), confirming high A + low B + moderate C as the productive recipe. Optimal mix: A > 0.45, B < 0.02, C ≈ 0.49–0.56.
- **F4 (4D, warehouse)**: Massive volatility (−33.24 to +0.303). The wk23 query `[0.354, 0.393, 0.352, 0.439]` returned −0.077 — slightly negative despite being inside the safe cluster. This confirms the landscape has razor-sharp cliffs; even small perturbations within the [0.35–0.45] zone can cross from positive to negative territory.
- **F5 (4D, chemical yield)**: Unimodal peak confirmed in the upper corner. Best output grew from ~1089 (initial) to 6117 (wk15). The wk23 near-zero corner probe `[0.070, 0.007, 0.002, 0.041]` returned only 163 — 37× lower than the peak — definitively confirming all four parameters must exceed ~0.85 for competitive yields.
- **F6 (5D, cake recipe)**: Best at −0.337 (wk17) with high butter (~0.99), moderate eggs (~0.62), low milk (~0.19). The wk23 probe with very low flour (0.031) returned −1.12, confirming flour is a critical positive driver (optimal ~0.35–0.40). Emerging recipe: moderate flour (0.35–0.40), moderate sugar/eggs (0.37–0.62), high butter (>0.99), low milk (<0.19).
- **F7 (6D, ML hyperparameters)**: Strong improvement trajectory from wk19–wk22 (outputs 2.34→2.57→2.79→2.65). Best at wk20 (2.791) with very low HP1 (~0.02). The wk23 query deliberately re-submitted the wk22 input and returned the identical output (2.652), confirming the function is deterministic and noise-free. The productive subspace is approximately `[0.02–0.15, 0.15–0.24, 0.33–0.47, 0.28–0.41, 0.21–0.35, 0.64–0.72]`.
- **F8 (8D, black-box)**: Peak at wk18 (9.895) with high P5 (0.951) and P8 (0.914). The wk23 query tested high P6 (0.989) with near-zero P5 (0.009) and low P8 (0.073), returning 9.315 (4th best). This reveals P6 as a previously underestimated driver — achieving 94% of the peak output without the two previously-assumed dominant parameters. The function may have multiple high-performance pathways through the 8D space.

---

## 5. Assumptions and Limitations

### Assumptions
1. **Smoothness**: All surrogates assume the objective function is locally smooth. If the function is discontinuous or has sharp ridges finer than the GP length-scale, the GP will under-fit.
2. **Stationarity**: The RBF kernel assumes the smoothness is constant across the domain. Non-stationary functions (e.g., with a very sharp peak in one region and a flat plateau elsewhere) will be mis-specified.
3. **Independence across functions**: Each of the eight functions is optimised independently. No transfer learning or multi-task GP is used.
4. **Query feedback is exact**: Outputs are treated as noise-free or low-noise. The WhiteKernel accounts for small noise but the pipeline is not designed for high-noise oracles.
5. **Uniform candidate grid**: For high-dim functions (F7–F8), candidates are uniform random draws. If the true optimum is in a very small region, it may not be sampled as a candidate and will be missed.

### Limitations and failure modes
- **Small dataset size**: With 21–51 observations, bootstrap ensemble variance estimates are noisy. The GP is the most reliable uncertainty model at this scale.
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
