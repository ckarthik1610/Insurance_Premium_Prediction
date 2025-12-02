# Home Insurance Sub-Package

This sub-package focuses on **home insurance premium prediction**. It has
two modules: `risk_factor.py` and `premium_calculator.py`.

## Module 1 – `risk_factor.py` (property_features.py)

**Purpose:**  
Extract and process structural, environmental, and geographical factors
that affect home insurance pricing.

### Main Functions

- `calculate_property_risk(features)`
  - Input: `PropertyFeatures` dataclass.
  - Output: Structural risk score in `[0, 1]`.
  - Uses property age, construction material, renovation status,
    security system, and number of floors.

- `environmental_risk_score(env)`
  - Input: dictionary with flood, wildfire, and crime indices.
  - Output: Environmental risk score in `[0, 1]`.

- `summarize_home_profile(features, env)`
  - Combines structural and environmental scores into a single **risk index**.
  - Used as an input feature for the premium model.

## Module 2 – `premium_calculator.py` (premium_estimator.py)

**Purpose:**  
Predict home insurance premiums using ML models built on the risk index,
property value, and claim history.

### Inheritance

- `BasePremiumModel`
  - Generic wrapper around a regression model (Random Forest).
- `HomePremiumEstimator(BasePremiumModel)`
  - Child class specialized for home insurance.
  - Builds feature matrices from property/environment information.

### Main Functions

- `train_home_model(records, risk_weight=1.0)`
  - Fits the `HomePremiumEstimator` based on training data.
- `predict_home_premium(model, features, env, property_value, num_past_claims)`
  - Returns the estimated annual premium for a specific home.
- `model_interpretation(model)`
  - Returns feature importance values to explain predictions.

## Example

See `examples/demo_home_insurance.py` for code that:
1. Builds a small training dataset.
2. Trains a home insurance premium model.
3. Predicts the premium for a new home.
4. Prints interpretation of feature importance.
