# BBO Capstone Project

## Documentation

| Document | Description |
|---|---|
| [DATASHEET.md](DATASHEET.md) | Dataset documentation following the Gebru et al. (2021) datasheets framework — motivation, composition, collection process, preprocessing, and distribution. |
| [MODEL_CARD.md](MODEL_CARD.md) | Model card following Mitchell et al. (2019) — approach overview, intended use, ten-round strategy details, performance results, assumptions, limitations, and ethical considerations. |

---

## Section 1: Project Overview
The BBO (Black-Box Optimization) capstone project focuses on optimizing an unknown function using limited queries. The purpose of this project is to develop and refine strategies for efficiently exploring and exploiting the search space to achieve optimal results. 

The overall goal of the BBO capstone project is to maximize an unknown objective function while adhering to constraints such as limited queries. This is highly relevant in real-world machine learning applications, such as hyperparameter tuning, experimental design, and resource allocation, where the underlying function is expensive or time-consuming to evaluate.

This project supports my career by enhancing my understanding of optimization techniques, improving my ability to balance exploration and exploitation, and deepening my knowledge of machine learning methods like Bayesian optimization, surrogate modeling, and heuristic approaches. These skills are valuable in solving complex, real-world problems in data science.

---

## Section 2: Inputs and Outputs
### Inputs:
- **Query Format**: The model receives a set of input parameters, represented as a vector of continuous values within a defined range (e.g., `[0, 1]` for normalized inputs).
- **Dimensions**: The input space may have multiple dimensions, depending on the problem (e.g., 2D, 3D, or higher) mentioned for each function. Dimensions are different for different functions.
- **Constraints**: Inputs must adhere to specific bounds, and the number of queries is limited.

### Outputs:
- **Response Value**: The model returns a scalar value representing the evaluation of the unknown function at the queried input. This could be a performance signal or best combination of ingrediants to make cake or.
- **Example**:
  - Input: `[0.5, 0.8]`
  - Output: `0.87` (objective function value)

---

## Section 3: Challenge Objectives
The primary objective of the BBO capstone project is to optimize the unknown function by maximizing its output. Key considerations include:
- **Goal**: Maximize the function's output within the given constraints.
- **Constraints**:
  - Limited number of queries to the function.
  - Unknown structure of the function (e.g., non-linear, noisy, or discontinuous).

The challenge lies in balancing exploration (searching for new areas of the input space) and exploitation (focusing on promising regions) to achieve the best possible result within the constraints.

---

## Section 4: Technical Approach
### Query Submission Strategies:
1. **Week 1 Submission**:
   - **Approach**: Random sampling across the input space to gather initial data and understand the function's behavior. Use Gaussian Process (GP) regression to model the unknown function and predict its behavior. Employ an acquisition function (e.g., Upper Confidence Bound ) to guide the next queries.
   - **Rationale**: Ensures broad coverage of the input space and avoids premature convergence.

2. **Second Submission**:
   - **Approach**: Refine the GP model with additional data and focus on exploitation/exploration by querying regions evaluate if needed to change the the strategy to explore or exploit depending on the functions behaviour.
   - **Rationale**: Balances exploration and exploitation by leveraging the GP model's uncertainty estimates.

3. **Third Submission**:
   - **Approach**: Refine the GP model with additional data and focus on exploitation/exploration by querying regions evaluate if needed to change the the strategy to explore or exploit depending on the functions behaviour. Analyze the use of SVM in the functions.
   - **Rationale**: Exploits promising areas identified in earlier submissions while still accounting for uncertainty.

4. **Fourth Submission**:
   - **Approach**: Refine the GP model with additional data and focus on exploitation/exploration by querying regions evaluate if needed to change the the strategy to explore or exploit depending on the functions behaviour. Analyze the use of Neural networks as a surrogate model in the functions.
   Try the categorize in such a way thet we could supervised learning methodology and copare if could use neural networks.
   - **Rationale**: Exploits promising areas identified in earlier submissions while still accounting for uncertainty.

5. **Fifth Submission**:
   - **Approach**: Refine the GP model with additional data and focus on exploitation/exploration by querying regions evaluate if needed to change the the strategy to explore or exploit depending on the functions behaviour. Compare Neural networks strategies with evaluating BBO problem.
   - **Rationale**: Do some more exploration to see behavious for more untested regions to check the behaviour.

6. ### Sixth Submission
- **Approach**: Move to a comparison of scalable surrogates: train a neural‑network surrogate (MC‑dropout or small ensemble) alongside an SVR ensemble and the existing GP where feasible. Use a mixed acquisition strategy (UCB + Expected Improvement) with a tuned beta to balance exploration/exploitation. Run Optuna-style tuning (multi‑fidelity / pruning) for key NN hyperparameters (lr, dropout, units) on a validation split. Perform batch scoring over X_grid to pick the next query.
- **Rationale**: NNs scale to higher dimensions and can leverage accumulated data; SVR ensembles provide a fast, robust baseline and empirical uncertainty. Mixed acquisition reduces risk from overconfident single‑model predictions. Hyperparameter tuning and pruning preserve the limited query budget.
- **Concrete actions**:
  - Train NN surrogate with MC‑dropout, and a bootstrap SVR ensemble; record mu and sigma per model.
  - Compute UCB_nn = mu_nn + beta * sigma_nn and UCB_svr = mu_svr + beta * sigma_svr; form combined score (weighted average or max) to select x_next.
  - Calibrate probabilities / uncertainties (calibration plots, Brier score) before relying on acquisition.
  - Query the black‑box at x_next, append to dataset, retrain surrogates, and log results (value, model used, uncertainty).
  - Save checkpoints, scalers and Optuna study results to experiments/ for reproducibility.

7. ### Seventh Submission — Results (suggested content)

- Summary: run length = 7 queries per function; batch scoring used; mixed acquisition (UCB + EI); surrogates compared = GP / SVR ensemble / NN (MC‑dropout).

- Key outcomes (fill with your values):
  - Best objective per function (value @ query #): 
    - F1: best = ____ (q=__); surrogate used = ____
    - F2: best = ____ (q=__); surrogate used = ____
    - F3: ...
  - Aggregate: median improvement over baseline = ____%; best single improvement = ____ (function __).
  - Calibration: Brier score (NN) = ____, Brier (GP) = ____, calibration slope/intercept = ____.
  - Optuna tuning: best val_loss = ____, best params = {lr:__, dropout:__, units:__}.
  - Runtime: average training time per iteration = __ sec; batch scoring cost = __ sec.

- Recommended plots to include in results/:
  - Best-so-far vs queries (one plot per function or combined small multiples).
  - Surrogate mean ± std heatmap or contour over X_grid (2D functions) for each surrogate.
  - Calibration plots (predicted prob vs empirical) and Brier scores.
  - Acquisition map (UCB/EI) showing selected x_next.
  - Optuna hyperparameter importance chart and study trace.

- Short interpretation (example bullets to adapt):
  - NN (MC‑dropout) outperformed SVR in high‑dim functions (F7/F8) after week 5, likely due to higher capacity and more data.
  - GP remained most reliable for low‑dim, low‑data functions (F1–F4) with best calibrated uncertainties.
  - Over‑exploitation observed on F5 after q=4 — reduce beta or increase exploration in next round.
  - Feature analysis: input 4 appears low‑importance for F5; consider dropping or reweighting.

- Next steps
  - Run ablations: (1) remove uncertainty, (2) increase beta, (3) ensemble top‑3 NNs — compare.
  - Save final surrogate checkpoints, scalers and Optuna study files to experiments/ and record seed for reproducibility.
  - Prepare a short "Results" notebook that loads experiments/ and reproduces the above plots for the report.

### Eighth Submission

- **Approach**: Finalise and evaluate the production‑ready pipeline. Build an ensemble surrogate (weighted combination of GP where feasible, SVR ensemble and NN ensemble / MC‑dropout). Use calibrated uncertainty to drive a small batch of final queries (batch BO) and run focused exploit/explore trade‑offs per function.

- **Rationale**: Combine strengths — GP calibration for low‑dim tasks, SVR for robust mid‑range performance, and NNs for scalability in higher dimensions. Ensemble + calibration reduces single‑model failure modes before final decisions.

- **Concrete actions**:
  - Produce final ensembles: train top‑K models per surrogate class, calibrate (temperature scaling / isotonic / Platt), and compute ensemble mu and sigma.
  - Run batch acquisition (3–5 points) per function using combined acquisition (weighted UCB/EI) and record outcomes.
  - Perform ablations: single model vs ensemble, different beta values, and different focus radii to justify choices.
  - Evaluate on held‑out temporal folds: best‑so‑far vs queries, calibration (Brier, reliability plots), stability across seeds.
  - Document final hyperparameters, Optuna study artifacts and checkpoints in experiments/; save summary plots under results/.

- **Evaluation & deliverables**:
  - Deliver a Results notebook reproducing final plots and metrics, an Experiments folder with configs + Optuna logs, and a short Results summary (one page) describing why the final pipeline was chosen.
  - Include deployment notes: recommended surrogate per data regime, batch scoring cadence, compute/latency estimates and monitoring checklist (drift, calibration, KPI alarms).

- **Next steps**: prepare a short demo notebook that runs one full BBO loop with the final pipeline and add a short section in README explaining when to prefer GP / SVR / NN depending on dimensionality and data volume.

### Ninth Submission (Weeks 21+)

- **Approach**: Use the fully assembled multi-surrogate pipeline established in submission 8 to generate and submit the next batch of query points across all eight functions. For each function, the three acquisition signals (GP-UCB, SVR-ensemble UCB, MLP-ensemble UCB) are compared and the point with the highest ensemble-consensus score is submitted. For signal-sparse functions (F1), the classification-based mixed acquisition (SVC probability + normalised predictive entropy) is run in parallel as a tie-breaker.

- **Surrogate stack (no TensorFlow required — all sklearn)**:
  - **GP** (`RBF + WhiteKernel`, `normalize_y=True`, 10 restarts) — primary surrogate for low-dim functions (F1–F4).
  - **SVR bootstrap ensemble** (25 models, `RBF`, `C=10`, `epsilon=1e-3`) — robust mid-range baseline with empirical uncertainty via bootstrap variance.
  - **MLP bootstrap ensemble** (50 models, `hidden=(64,64)`, `relu`, `max_iter=500`) — scalable surrogate replacing MC-Dropout; used as primary for high-dim functions (F7–F8).
  - **Classification layer** (`LogisticRegression` + `SVC(probability=True)` + `MLPClassifier(32,16)`) — binary labels at 75th-percentile threshold; mixed acquisition = class probability + normalised predictive entropy.

- **Acquisition strategy**:
  - Low-dim (2D, F1–F2): GP-UCB with β = 30 (exploration-heavy) on a 100×100 grid; MLP-ensemble UCB with β = 6 on the same grid; SVR-ensemble UCB with β = 6.
  - High-dim (8D, F7–F8): GP-UCB with κ = 10 over 50,000 random candidates.
  - Final point selection: argmax of the ensemble-averaged UCB score across all three surrogates.

- **Key findings from submission 8 data (used to inform submission 9 queries)**:
  - **F1 (2D, radiation)**: Most observed outputs are near zero; the strongest signal observed was −0.00765 (wk13). The GP mean map and classification acquisition both point to the region around `[0.65, 0.68]` as the most informative next zone.
  - **F8 (8D, hyperparameter tuning)**: Best observed output = **9.8948** at `[0.014, 0.203, 0.064, 0.132, 0.951, 0.485, 0.038, 0.914]` (wk18). GP-UCB with κ = 10 over 50k candidates drives the next suggested point toward under-sampled high-uncertainty regions.

- **Rationale**: With 20+ observations per function, the surrogates have sufficient data to reduce uncertainty in promising regions. Submission 9 shifts the balance slightly toward exploitation (lower β/κ on functions with a clear best candidate) while maintaining exploration on functions still showing flat or sparse signal.

- **Next steps after submission 9**:
  - Compare best-so-far curves across all 8 functions to identify which surrogates drove the most improvement.
  - Plot GP mean ± σ heatmaps (2D) and pairwise scatter matrices coloured by score (8D) to visualise progress.
  - Evaluate whether the MLP ensemble uncertainty is well-calibrated (compare predicted σ vs actual residuals).
  - Prepare a final Results notebook consolidating all query histories, surrogate comparisons, and best observed values per function.

### Tenth Submission (Week 21)

- **Approach**: This round was defined by a single lesson from round nine: step sizes matter more than direction. Conservative near-repeats consistently outperformed bold gradient-following perturbations. The focus radius was halved on every surrogate-driven function (from 0.025–0.04 to 0.01–0.015), and per-function acquisition parameters were tuned based on nine rounds of accumulated evidence.

- **Surrogate stack (unchanged from submission 9 — all sklearn, no TensorFlow)**:
  - **GP** (`RBF + WhiteKernel`, `normalize_y=True`, 10 restarts) — primary surrogate for all functions.
  - **SVR bootstrap ensemble** (25 models, `RBF`, `C=10`, `epsilon=1e-3`) — robust baseline with empirical uncertainty (F1).
  - **MLP bootstrap ensemble** (50 models, `hidden=(64,64)`, `relu`, `max_iter=500`) — scalable surrogate (F1).
  - **Classification layer** (`LogisticRegression` + `SVC(probability=True)` + `MLPClassifier(32,16)`) — entropy-based acquisition for signal-sparse F1.

- **Per-function acquisition strategy**:

  | Function | Dim | κ / β | Candidates | Method |
  |---|---|---|---|---|
  | F1 | 2D | β = 6 (GP, SVR, MLP); entropy (classification) | 100×100 grid | Multi-surrogate + classification |
  | F2 | 2D | κ = 3 | 100×100 grid | GP-UCB |
  | F3 | 3D | κ = 30 | 5,000 random | GP-UCB (high exploration) |
  | F4 | 4D | κ = 3 | 5,000 random | GP-UCB |
  | F5 | 4D | κ = 4 | 5,000 random | GP-UCB |
  | F6 | 5D | κ = 4 | 20,000 random | GP-UCB |
  | F7 | 6D | κ = 3 | 20,000 random | GP-UCB |
  | F8 | 8D | κ = 10 | 50,000 random | GP-UCB (aggressive exploration) |

### Eleventh Submission (Week 22) — Clustering-Informed Refinement

- **Approach**: This round viewed the accumulated search space through a clustering lens — identifying natural groupings in past queries, measuring inter-cluster distances, and using centroid trends to guide queries. The goal was to tighten exploration around high-performing clusters while avoiding the noise regions that earlier rounds had already ruled out. Each query targets a specific local cluster, guided by the GP-UCB acquisition function with kappa values tuned per function based on cluster spread.

- **Surrogate stack (unchanged from submission 10 — all sklearn, no TensorFlow)**:
  - **GP** (`RBF + WhiteKernel`, `normalize_y=True`, 10 restarts) — primary surrogate for all functions.
  - **SVR bootstrap ensemble** (25 models, `RBF`, `C=10`, `epsilon=1e-3`) — robust baseline with empirical uncertainty (F1).
  - **MLP bootstrap ensemble** (50 models, `hidden=(64,64)`, `relu`, `max_iter=500`) — scalable surrogate (F1).
  - **Classification layer** (`LogisticRegression` + `SVC(probability=True)` + `MLPClassifier(32,16)`) — entropy-based acquisition for signal-sparse F1.

- **Per-function query rationale (wk22)**:

  | Function | wk22 input | wk22 output | Cluster / strategy |
  |---|---|---|---|
  | F1 | `[1.000000, 1.000000]` | 0.0 | Random corner probe — no signal detected in 9 prior rounds; maximise coverage in untested extremes |
  | F2 | `[0.868686, 0.303030]` | 0.503 | On the high-x1 ridge near best (wk20: 0.697); tightening along the identified band |
  | F3 | `[0.979416, 0.018415, 0.044478]` | −0.128 | High A, low B+C corner — testing if the best-region (low B) extends to extreme values |
  | F4 | `[0.390608, 0.391840, 0.339213, 0.408180]` | 0.153 | Inside the tight [0.35–0.45] cluster centroid; radius ~0.01 from the best (wk20: 0.303) |
  | F5 | `[0.943487, 0.987941, 0.947113, 0.913539]` | 5392.4 | Upper-corner cluster exploitation; all params >0.91, near the peak zone |
  | F6 | `[0.201604, 0.945689, 0.965562, 0.984333, 0.974845]` | −2.185 | Exploratory probe with high sugar+eggs+butter+milk — confirmed that high milk (0.97) degrades performance |
  | F7 | `[0.149979, 0.153680, 0.445210, 0.316094, 0.205053, 0.716571]` | 2.652 | Extending the improving trajectory (wk19–21: 2.34→2.57→2.79); second-best output confirms the cluster direction |
  | F8 | `[0.268584, 0.149940, 0.020668, 0.001506, 0.086266, 0.040657, 0.088428, 0.926627]` | 9.384 | Exploring low-param region variant; high P8 (0.927) retained from the best cluster; third-best overall |

### Twelfth Submission (Week 23) — Boundary Probing and Validation

- **Approach**: This round systematically probed the boundaries of known productive regions and ran validation queries. The goal was twofold: (1) test whether the identified optima are truly local or whether the productive subspaces extend further than assumed, and (2) verify function determinism by re-querying a known point. Several functions received deliberate extreme-corner or boundary probes to map the edge of the response surface. Exploration kappa values were increased for key functions (F4: κ = 4; F5: κ = 8) to encourage broader boundary testing.

- **Surrogate stack (unchanged from submission 11 — all sklearn, no TensorFlow)**:
  - **GP** (`RBF + WhiteKernel`, `normalize_y=True`, 10 restarts) — primary surrogate for all functions.
  - **SVR bootstrap ensemble** (25 models, `RBF`, `C=10`, `epsilon=1e-3`) — robust baseline with empirical uncertainty (F1).
  - **MLP bootstrap ensemble** (50 models, `hidden=(64,64)`, `relu`, `max_iter=500`) — scalable surrogate (F1).
  - **Classification layer** (`LogisticRegression` + `SVC(probability=True)` + `MLPClassifier(32,16)`) — entropy-based acquisition for signal-sparse F1.

- **Per-function query rationale (wk23)**:

  | Function | wk23 input | wk23 output | Strategy / finding |
  |---|---|---|---|
  | F1 | `[0.494949, 0.404040]` | −8.17e-5 | South-west probe away from known signal zone `[0.65, 0.68]`; third-strongest signal ever detected (after wk13: −7.65e-3 and wk14: −1.27e-4) — first non-trivial response since wk15, suggesting the non-zero support may extend further than assumed |
  | F2 | `[0.969696, 0.373737]` | −0.053 | Extreme high-x1 boundary probe; first negative output in student queries — confirms the productive ridge drops off sharply beyond x1 ≈ 0.87 |
  | F3 | `[0.977393, 0.006965, 0.491366]` | −0.077 | High A, very low B, moderate C; second-best student query after wk13 (−0.056) — confirms high A + low B + moderate C is productive |
  | F4 | `[0.354274, 0.393257, 0.352035, 0.438677]` | −0.077 | Inside the safe cluster but slightly below centre; turned negative (vs +0.153 at wk22), confirming extreme landscape sensitivity |
  | F5 | `[0.070204, 0.006602, 0.002019, 0.040583]` | 163.0 | Near-zero corner boundary probe; output 37× lower than peak (6117), confirming all params must be >0.85 for competitive yields |
  | F6 | `[0.031407, 0.712762, 0.466143, 0.989009, 0.063110]` | −1.120 | Very low flour (0.031) with high butter (0.989) and low milk (0.063); output 3.3× worse than best — confirms flour must not be too low (optimal ~0.40) |
  | F7 | `[0.149979, 0.153680, 0.445210, 0.316094, 0.205053, 0.716571]` | 2.652 | Deliberate re-query of wk22 input; returned identical output (2.652), confirming the function is deterministic and noise-free |
  | F8 | `[0.179766, 0.235992, 0.241664, 0.009754, 0.008830, 0.989489, 0.083388, 0.073262]` | 9.315 | High P6 (0.989) but near-zero P5 (0.009) and low P8 (0.073); 4th-best output — reveals P6 as a previously underestimated driver alongside P5 and P8 |

### Thirteenth Submission (Week 24) — Exploitation Refinement and Extremes Testing

- **Approach**: This round combined targeted exploitation near known optima (F7) with deliberate extreme-parameter probes (F3, F5, F6, F8) to further map boundary conditions. The surrogate stack and acquisition parameters remain unchanged from submission 12. Key goals: (1) attempt to surpass the F7 best by perturbing the productive subspace, (2) test whether extreme Compound C (F3) or extreme sugar/eggs (F6) configurations are viable, (3) continue characterising the multi-pathway structure of F8.

- **Surrogate stack (unchanged from submission 12 — all sklearn, no TensorFlow)**:
  - **GP** (`RBF + WhiteKernel`, `normalize_y=True`, 10 restarts) — primary surrogate for all functions.
  - **SVR bootstrap ensemble** (25 models, `RBF`, `C=10`, `epsilon=1e-3`) — robust baseline with empirical uncertainty (F1).
  - **MLP bootstrap ensemble** (50 models, `hidden=(64,64)`, `relu`, `max_iter=500`) — scalable surrogate (F1).
  - **Classification layer** (`LogisticRegression` + `SVC(probability=True)` + `MLPClassifier(32,16)`) — entropy-based acquisition for signal-sparse F1.

- **Per-function query rationale (wk24)**:

  | Function | wk24 input | wk24 output | Strategy / finding |
  |---|---|---|---|
  | F1 | `[0.565656, 1.000000]` | 0 | North-edge probe at x2=1.0; returned zero, confirming the non-zero support does not extend to the upper boundary. The signal tail detected at wk23 `[0.49, 0.40]` does not continue northward |
  | F2 | `[0.565656, 0.191919]` | 0.102 | Low-x2 probe; weak positive output (7× below best 0.697). Confirms the productive ridge requires x2 ≈ 0.30–0.40 and x1 > 0.85 — this query was too far south and west |
  | F3 | `[0.397631, 0.012935, 0.995075]` | −0.508 | Extreme high-C probe with moderate A and low B; **worst student output** — 9× worse than best (−0.056). Very high Compound C (>0.99) is strongly detrimental; optimal C remains ≈ 0.49–0.56 |
  | F4 | `[0.461404, 0.409086, 0.352151, 0.423373]` | −0.163 | Slightly outside the safe cluster centre (HP1=0.461 vs optimal ~0.40); negative output confirms the narrow viable zone. HP1 > 0.45 appears to cross the landscape cliff |
  | F5 | `[0.997964, 0.986498, 0.096838, 0.830882]` | 2640 | Two high params (P1≈1.0, P2≈0.99) but P3 low (0.097); output 2.3× below peak — confirms **all four** parameters must be >0.85, not just most of them |
  | F6 | `[0.872310, 0.043940, 0.030514, 0.995411, 0.993058]` | −2.526 | High flour (0.87) and high butter (0.995) but near-zero sugar (0.044) and eggs (0.031) with high milk (0.993); 7.5× worse than best — confirms sugar and eggs are also critical positive drivers, and high milk remains strongly detrimental |
  | F7 | `[0.218524, 0.220790, 0.480522, 0.354701, 0.354809, 0.596844]` | **2.908** | **NEW BEST** — surpasses wk20's 2.791 by 4.2%. Slightly higher HP1 (0.22 vs 0.02), similar HP3/HP4, lower HP6 (0.60 vs 0.64). Confirms the productive subspace is broader than assumed |
  | F8 | `[0.009290, 0.980500, 0.084033, 0.872849, 0.035300, 0.959276, 0.976142, 0.034148]` | 7.025 | High P2+P4+P6+P7 pathway test; lowest output in recent rounds. Unlike the P6-only success at wk23 (9.315), this high-P7 configuration significantly underperformed — P7 appears to be a negative or neutral driver |

- **Final best outputs per function (after all 12 rounds, wk13–wk24)**:

  | Function | Scenario | Best output | Best input | Best week |
  |---|---|---|---|---|
  | F1 | 2D contamination detection | −0.00765 | `[0.646, 0.677]` | wk13 |
  | F2 | 2D noisy log-likelihood | **0.697** | `[0.859, 0.343]` | wk20 |
  | F3 | 3D drug compound minimisation | **−0.056** | `[0.448, 0.218, 0.560]` | wk13 |
  | F4 | 4D warehouse tuning | **0.303** | `[0.404, 0.434, 0.436, 0.384]` | wk20 |
  | F5 | 4D chemical yield | **6117.3** | `[0.886, 0.998, 0.960, 0.993]` | wk15 |
  | F6 | 5D cake recipe | **−0.337** | `[0.399, 0.372, 0.622, 0.993, 0.189]` | wk17 |
  | F7 | 6D ML hyperparameters | **2.908** | `[0.219, 0.221, 0.481, 0.355, 0.355, 0.597]` | wk24 |
  | F8 | 8D black-box | **9.895** | `[0.014, 0.203, 0.064, 0.132, 0.951, 0.485, 0.038, 0.914]` | wk18 |

- **Key findings across all twelve rounds (wk13–wk24)**:
  - **F1 (2D, radiation)**: Signal is extremely localised. Outputs collapsed to effectively zero from wk16–wk22, with a brief non-trivial signal at wk23 (−8.17e-5 at `[0.49, 0.40]`). The wk24 north-edge probe `[0.566, 1.0]` returned zero, confirming the signal tail does not extend to the upper boundary. The non-zero support remains concentrated around `[0.65, 0.68]` with a possible low-amplitude tail toward the south-west.
  - **F2 (2D, log-likelihood)**: Best remains 0.697 at wk20. The wk24 probe at `[0.566, 0.192]` returned 0.102 — weak but positive. Both wk23 (x1=0.97, negative) and wk24 (x1=0.57, weak) confirm the productive zone is narrow: x1 ≈ 0.85–0.87, x2 ≈ 0.30–0.40.
  - **F3 (3D, drug compounds)**: Best output (−0.056) still from wk13. The wk24 extreme high-C probe `[0.398, 0.013, 0.995]` returned −0.508 — the worst student result, 9× worse than best. This definitively establishes that Compound C must stay moderate (0.49–0.56); extreme values in either direction are detrimental.
  - **F4 (4D, warehouse)**: Best at +0.303 (wk20). The wk24 query `[0.461, 0.409, 0.352, 0.423]` returned −0.163 — negative despite being near the cluster. HP1 = 0.461 appears to cross the landscape cliff (optimal HP1 ≈ 0.39–0.44). The viable zone is extremely narrow.
  - **F5 (4D, chemical yield)**: Best 6117 at wk15. The wk24 query with P3=0.097 returned only 2640 — confirming that even one low parameter (out of four) halves the yield. All four parameters must simultaneously exceed ~0.85.
  - **F6 (5D, cake recipe)**: Best at −0.337 (wk17). The wk24 probe with near-zero sugar (0.044) and eggs (0.031) but high flour (0.87) and butter (0.995) returned −2.526. This reveals that sugar and eggs are also critical positive drivers, not just flour and butter. High milk (0.993) continues to be strongly detrimental. Optimal recipe requires balanced moderate-to-high flour (0.35–0.40), sugar (>0.37), eggs (>0.37), butter (>0.99), and very low milk (<0.19).
  - **F7 (6D, ML hyperparameters)**: **New best at wk24: 2.908** at `[0.219, 0.221, 0.481, 0.355, 0.355, 0.597]`, surpassing wk20's 2.791 by 4.2%. The winning input has higher HP1 (0.22 vs 0.02) and lower HP6 (0.60 vs 0.64) than the previous best — the productive subspace is broader than previously assumed, approximately `[0.02–0.22, 0.15–0.24, 0.33–0.48, 0.28–0.41, 0.21–0.36, 0.60–0.72]`.
  - **F8 (8D, black-box)**: Peak at wk18 (9.895). The wk24 high-P2+P4+P6+P7 test returned only 7.025 — the lowest in recent rounds. High P7 (0.976) appears detrimental compared to the wk23 result where high P6 alone (with low P7) achieved 9.315. P5 and P8 remain the strongest drivers; P6 is a secondary driver; P7 is likely neutral or negative.

- **Lessons learned (accumulated across all 12 rounds)**:
  - Step sizes matter more than direction: conservative near-repeats consistently outperformed bold gradient-following perturbations across all functions.
  - Higher dimensions demand more candidates (50k for 8D) and higher κ (exploration), while lower dimensions benefit from grid-based exploitation.
  - The GP surrogate with RBF kernel performed reliably for all dimensionalities when paired with appropriate κ scheduling. Ensemble surrogates (SVR, MLP) added value primarily for uncertainty quantification on signal-sparse functions (F1).
  - Exploratory probes into untested regions carried high downside risk (F4: −33.24; F5: 103; F6: −2.75) but were essential for identifying landscape boundaries.
  - Boundary probing in round 12 revealed that productive regions can have unexpected tails (F1 signal at `[0.49, 0.40]`) and that assumed dominant parameters may not be the only drivers (F8: P6 nearly as impactful as P5+P8).
  - Reproducibility verification (F7 re-query) confirmed deterministic function behaviour, validating that observed noise in other functions is genuine landscape roughness rather than evaluation noise.
  - Round 13 delivered a new F7 best (2.908 vs 2.791) by exploring a slightly broader subspace — confirming that productive regions can be wider than initial convergence suggests, and that continued exploitation with small perturbations remains valuable even in later rounds.
  - Extreme single-parameter probes (F3: C=0.995, F6: sugar≈0, eggs≈0) consistently produced worst-case outputs, establishing clear boundary constraints for each function.

---

### Machine Learning Methods:
- **Gaussian Processes**: Used to model the unknown function and predict outputs based on prior observations.
- **Acquisition Functions**: Guides the selection of new queries by balancing exploration and exploitation.
- **Bayesian Optimization**: Provides a structured framework for optimizing the function under uncertainty.

### Exploration vs. Exploitation:
- **Exploration**: Random sampling and uncertainty-based acquisition functions ensure that the model explores untested regions of the input space.
- **Exploitation**: Focuses on regions with high predicted values to maximize the objective function.

### Unique Aspects:
- Thoughtful use of Bayesian optimization to adaptively refine the search space.
- Iterative improvement of the surrogate model to better approximate the unknown function.
- Careful balance of exploration and exploitation to maximize performance within the query constraints.
