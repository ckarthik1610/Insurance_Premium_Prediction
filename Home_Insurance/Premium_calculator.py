import pandas as pd
import pickle
import os

from Home_Insurance.Risk_factor import data 

class ModelFileNotFoundError(Exception):
    pass

class FeatureFileNotFoundError(Exception):
    pass

class Predict(data):
    def __init__(self, model_path, feature_path):
        
        if not os.path.exists(model_path):
            raise ModelFileNotFoundError(
                f"Model file not found at: {model_path}"
            )

        # --- Check feature file ---
        if not os.path.exists(feature_path):
            raise FeatureFileNotFoundError(
                f"Feature file not found at: {feature_path}"
            )
        data.__init__(self, model_path)

        self.model_path = model_path
        self.feature_path = feature_path

        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

        with open(self.feature_path, "rb") as f:
            self.features = pickle.load(f)

    def predict_price(self, input_dict: dict) -> float:
        
        df = pd.DataFrame([input_dict])

        df = self.encoding(df)

        df = df.reindex(columns=self.features, fill_value=0)

        pred = self.model.predict(df)[0]

        return round(pred, 2)
    