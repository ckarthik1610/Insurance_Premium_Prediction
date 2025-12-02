"""
Module: premium_estimator.py

Predicts home insurance premiums using simple statistical / ML models
built on risk and property features.
"""

from typing import Iterable, List, Tuple
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .risk_factor import summarize_home_profile, PropertyFeatures


class BasePremiumModel:
    """
    Base class that wraps an underlying regression model.
    This class will be inherited by specific insurance models.
    """

    def __init__(self, model=None):
        self.model = model or RandomForestRegressor(
            n_estimators=100,
            random_state=42,
        )
        self.feature_names_: List[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BasePremiumModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class HomePremiumEstimator(BasePremiumModel):
    """
    Child class that specializes BasePremiumModel for
    home insurance premium estimation.  (INHERITANCE HERE)
    """

    def __init__(self, risk_weight: float = 1.0, model=None):
        super().__init__(model=model)
        self.risk_weight = risk_weight

    def build_feature_matrix(
        self,
        records: Iterable[Tuple[PropertyFeatures, dict, float, int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert raw input into X, y suitable for modelling.

        Each record is a tuple:
        (property_features, env_dict, property_value, num_past_claims, premium)

        property_value and num_past_claims are simple numeric predictors.
        """
        X_list = []
        y_list = []

        for prop_feat, env, property_value, num_claims, premium in records:
            risk_index = summarize_home_profile(prop_feat, env)

            X_list.append(
                [
                    property_value,
                    num_claims,
                    risk_index * self.risk_weight,
                ]
            )
            y_list.append(premium)

        self.feature_names_ = [
            "property_value",
            "num_past_claims",
            "risk_index",
        ]

        return np.asarray(X_list, dtype=float), np.asarray(y_list, dtype=float)




def train_home_model(
    records: Iterable[Tuple[PropertyFeatures, dict, float, int, float]],
    risk_weight: float = 1.0,
) -> HomePremiumEstimator:
    """
    Fits ML models based on property value, claims, and risk profiles.

    Parameters
    ----------
    records : iterable
        Each element: (PropertyFeatures, env_dict, property_value,
                       num_past_claims, premium)
    risk_weight : float
        Weight applied to the combined risk index.

    Returns
    -------
    HomePremiumEstimator
        Fitted estimator.
    """
    estimator = HomePremiumEstimator(risk_weight=risk_weight)
    X, y = estimator.build_feature_matrix(records)
    estimator.fit(X, y)
    return estimator


def predict_home_premium(
    model: HomePremiumEstimator,
    features: PropertyFeatures,
    env: dict,
    property_value: float,
    num_past_claims: int,
) -> float:
    """
    Produces estimated premium amounts for a given home description.

    Parameters
    ----------
    model : HomePremiumEstimator
    features : PropertyFeatures
    env : dict
    property_value : float
    num_past_claims : int

    Returns
    -------
    float
        Predicted annual premium.
    """
    risk_index = summarize_home_profile(features, env)
    X = np.asarray(
        [[property_value, num_past_claims, risk_index * model.risk_weight]],
        dtype=float,
    )
    return float(model.predict(X)[0])


def model_interpretation(model: HomePremiumEstimator) -> dict:
    """
    Generates simple feature-importance based interpretation
    to explain predictions.

    Returns
    -------
    dict
        Mapping feature_name -> importance (higher = more important).
    """
    importances = getattr(model.model, "feature_importances_", None)
    if importances is None:
        raise ValueError("Underlying model does not provide feature_importances_")

    return {
        name: float(imp)
        for name, imp in zip(model.feature_names_, importances)
    }
