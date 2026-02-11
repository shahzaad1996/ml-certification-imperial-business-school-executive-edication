# BBO Capstone Project

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
