from insurance_premium_package.home_insurance import (
    PropertyFeatures,
    calculate_property_risk,
    environmental_risk_score,
    summarize_home_profile,
    train_home_model,
    predict_home_premium,
    model_interpretation,
)


def build_dummy_training_data():
    """Create a tiny fake dataset just to show how the functions work."""
    records = []

    
    p1 = PropertyFeatures(5, "brick", 2020, True, 2)
    e1 = {"flood_zone_level": 0.1, "wildfire_risk": 0.0, "crime_rate_index": 0.3}
    records.append((p1, e1, 500_000, 0, 900))

    p2 = PropertyFeatures(25, "wood", 2010, False, 1)
    e2 = {"flood_zone_level": 0.6, "wildfire_risk": 0.1, "crime_rate_index": 0.5}
    records.append((p2, e2, 400_000, 1, 1400))

    p3 = PropertyFeatures(40, "brick", None, False, 3)
    e3 = {"flood_zone_level": 0.8, "wildfire_risk": 0.3, "crime_rate_index": 0.7}
    records.append((p3, e3, 650_000, 2, 2100))

    return records


def main():
   
    p_feat = PropertyFeatures(12, "brick", 2015, True, 2)
    env = {"flood_zone_level": 0.3, "wildfire_risk": 0.2, "crime_rate_index": 0.4}

    print("Property risk:", calculate_property_risk(p_feat))
    print("Environmental risk:", environmental_risk_score(env))
    print("Combined risk:", summarize_home_profile(p_feat, env))

   
    records = build_dummy_training_data()
    model = train_home_model(records)

   
    premium = predict_home_premium(
        model,
        features=p_feat,
        env=env,
        property_value=550_000,
        num_past_claims=1,
    )
    print("Predicted premium:", premium)

    # interpretation
    print("Feature importance:", model_interpretation(model))


if __name__ == "__main__":
    main()
