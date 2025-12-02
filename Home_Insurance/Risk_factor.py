"""
Module: property_features.py

Extracts and processes structural, environmental, and geographical
factors affecting home insurance pricing. Preprocesses the data
for prediction and creates risk scores.
"""

from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class PropertyFeatures:
    """
    Represents the structural characteristics of a home.
    """
    property_age: int                 # in years
    construction_material: str        # e.g. "brick", "wood"
    renovation_year: Optional[int]    # None if never renovated
    has_security_system: bool
    num_floors: int

    def age_score(self) -> float:
        """
        Converts property age into a simple risk score.
        Newer homes → lower risk, very old homes → higher risk.
        """
        if self.property_age < 10:
            return 0.2
        elif self.property_age < 30:
            return 0.5
        else:
            return 0.8


# --------- Functions required by your slide ----------

def calculate_property_risk(features: PropertyFeatures) -> float:
    """
    Determines structural risk using property age, materials,
    and renovation status.

    Parameters
    ----------
    features : PropertyFeatures
        Dataclass with information about the home.

    Returns
    -------
    float
        Property risk score between 0 and 1
        (higher = more risky).
    """
    score = features.age_score()

    # construction material effect
    if features.construction_material.lower() in {"brick", "concrete"}:
        score -= 0.1
    else:  # e.g. wood
        score += 0.1

    # renovation effect
    if features.renovation_year is not None and features.property_age > 20:
        # recently renovated older home → slightly less risk
        score -= 0.1

    # security system effect
    if features.has_security_system:
        score -= 0.1

    # floors – more floors = slightly more risk
    score += 0.02 * (features.num_floors - 1)

    return max(0.0, min(1.0, score))


def environmental_risk_score(env: Dict[str, float]) -> float:
    """
    Assesses environment-based risks such as flood zones or fire
    vulnerability.

    Parameters
    ----------
    env : dict
        Dictionary with keys like:
        - "flood_zone_level" : 0 (no risk) – 1 (very high)
        - "wildfire_risk"    : 0 – 1
        - "crime_rate_index" : 0 – 1

    Returns
    -------
    float
        Environmental risk score between 0 and 1.
    """
    flood = env.get("flood_zone_level", 0.0)
    fire = env.get("wildfire_risk", 0.0)
    crime = env.get("crime_rate_index", 0.0)

    
    score = 0.5 * flood + 0.3 * fire + 0.2 * crime
    return max(0.0, min(1.0, score))


def summarize_home_profile(
    features: PropertyFeatures,
    env: Dict[str, float],
) -> float:
    """
    Combines all property and environmental features into
    a unified risk index.

    Parameters
    ----------
    features : PropertyFeatures
    env : dict
        Environmental information used by `environmental_risk_score`.

    Returns
    -------
    float
        Combined risk index between 0 and 1.
    """
    prop_score = calculate_property_risk(features)
    env_score = environmental_risk_score(env)

   
    combined = 0.6 * prop_score + 0.4 * env_score
    return max(0.0, min(1.0, combined))
