"""
Home Insurance sub-package.

This sub-package contains functionality to:
- Extract and process property and environmental features.
- Compute risk scores.
- Train and interpret models to estimate home insurance premiums.
"""

from .risk_factor import (
    calculate_property_risk,
    environmental_risk_score,
    summarize_home_profile,
)

from .premium_calculator import (
    HomePremiumEstimator,
    train_home_model,
    predict_home_premium,
    model_interpretation,
)
