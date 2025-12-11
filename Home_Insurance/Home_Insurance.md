# Home Insurance Premium Prediction
The Home Insurance Premium Prediction system estimates the *Annual Home Insurance Premium Price* for a property given its risk factors and characteristics. It takes detailed property data as input and uses a *Linear Regression* model to predict the premium. The Home Insurance sub-package contains two modules:# Home Insurance Premium Prediction

The Home Insurance Premium Prediction system estimates the **Annual Home Insurance Premium Price** for a property given its risk factors and characteristics. It takes detailed property data as input and uses a **Linear Regression** model to predict the premium. The Home Insurance sub-package contains two modules:

* `Risk_factor.py`
* `Premium_calculator.py`

# Input and Output

### Input - Property Characteristics

* Claim_3_Years
* Owner
* Employment_Status
* Accidental_Damage
* Owner_Sex
* Alarm_Present
* Locks_Present
* Bedrooms
* Flooding
* Safe_Installed
* YearBuilt

### Output

* Estimated Annual Home Insurance Premium Price

# Module: `Risk_factor.py`

This module prepares the dataset, trains the Linear Regression model, and saves model artifacts. It contains the class `data`.

## Class: `data`

* `__init__(self, dir)`
  Initializes with the directory path to the source CSV (expected `dataset.csv`) and sets up attributes for data, features, model, etc.

* `preprocess(self)`
  Reads the dataset, separates predictors from the target column `Annual_Premium_Price`, and converts categorical predictors into numeric columns using **One-Hot Encoding** (`pd.get_dummies`) so the data is ready for the Linear Regression model.

* `train(self)`
  Initializes and fits a `sklearn.linear_model.LinearRegression` model on the preprocessed data. Stores the trained model (e.g., `self.model`) and the final feature list (`self.features`).

* `save(self)`
  Persists the trained model and the feature names list using `pickle`. Produces two files used by the predictor: `Linear_Regression.pkl` and `Feature_names.pkl`.

# Module: `Premium_calculator.py`

This module loads the trained model and computes final premium estimates for new property inputs. It contains the class `Predict`, which inherits from `data`.

## Class: `Predict`

* `__init__(self, model_path, feature_path)`
  Loads the saved Linear Regression model and the expected feature names list from their `.pkl` files using `pickle`. Prepares any structures needed to align new inputs with training features.

* `predict_price(self, input_dict: dict)`
  Accepts a dictionary of property inputs, converts it to a `pandas.DataFrame`, applies the same One-Hot Encoding strategy (so new categorical levels align with training features), reindexes/aligns columns to the saved feature list, then uses the loaded Linear Regression model to predict and return the **estimated Annual Premium Price** (rounded float).

# Requirements

* `pandas`
* `pickle`
* `scikit-learn` (specifically `sklearn.linear_model.LinearRegression`)

# Demo Output

![Home Insurance](/Output/Home_Insurance.png)

---

### Notes / Implementation tips

* Make sure to save the exact feature ordering used during training (e.g., `Feature_names.pkl`) so `Predict` can align new inputs correctly.
* When encoding new inputs, add missing columns (fill with 0) and ignore unknown columns introduced at prediction time so the feature vector always matches the saved feature list.
* Persist the trained model with `pickle` (or `joblib`) and keep versioning for reproducibility (e.g., include model training date or version string).
