# BBO Capstone Project

## Documentation

| Document | Description |
|---|---|
| [DATASHEET.md](DATASHEET.md) | Dataset documentation following the Gebru et al. (2021) datasheets framework — motivation, composition, collection process, preprocessing, and distribution. |
| [MODEL_CARD.md](MODEL_CARD.md) | Model card following Mitchell et al. (2019) — approach overview, intended use, nine-round strategy details, performance results, assumptions, limitations, and ethical considerations. |

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
