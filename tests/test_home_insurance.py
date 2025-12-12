import unittest
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from Home_Insurance.Premium_calculator import Predict


class TestHomePredict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("Setting up class resources...")

        cls.model_path = "Home_Insurance/test_home_model.pkl"
        cls.feature_path = "Home_Insurance/test_home_features.pkl"

        # ✅ Match your Risk_factor predictors (after encoding, before model)
        cls.features = [
            "Claim_3_Years",
            "Owner_Employment_Status",
            "Accidental_Damage",
            "Owner_Sex",
            "Alarm_Present",
            "Locks_Present",
            "Bedrooms",
            "Flooding",
            "Safe_Installed",
            "YearBuilt",
            # include if your dataset has it; harmless if you keep it consistent here + input_dict
            "Owner",
        ]

        X_train = np.zeros((5, len(cls.features)))
        y_train = np.full(5, 123.456)

        model = DummyRegressor(strategy="mean")
        model.fit(X_train, y_train)

        with open(cls.model_path, "wb") as f:
            pickle.dump(model, f)

        with open(cls.feature_path, "wb") as f:
            pickle.dump(cls.features, f)

    @classmethod
    def tearDownClass(cls):
        print("Cleaning up class resources...")
        if os.path.exists(cls.model_path):
            os.remove(cls.model_path)
        if os.path.exists(cls.feature_path):
            os.remove(cls.feature_path)

    def setUp(self):
        print("Setting Up")

        self.predictor = Predict(self.model_path, self.feature_path)

        # ✅ Use raw strings to test your encoding normalization
        self.input_dict = {
            "Claim_3_Years": "Yes",
            "Owner_Employment_Status": "no",   # lower-case (should still work)
            "Accidental_Damage": " YES ",      # extra spaces
            "Owner_Sex": "m",                  # lower-case
            "Alarm_Present": "No",
            "Locks_Present": "Yes",
            "Bedrooms": 3,
            "Flooding": "No",
            "Safe_Installed": "Yes",
            "YearBuilt": 2001,
            "Owner": 1,                        # numeric as-is (if you have it)
        }

        self.expected_price = round(123.456, 2)  # 123.46

    def tearDown(self):
        print("tearingDown")

    def test_predict_price_output(self):
        price = self.predictor.predict_price(self.input_dict)

        self.assertIsInstance(price, float)
        self.assertEqual(price, self.expected_price)
        self.assertGreater(price, 0)
        self.assertEqual(round(price, 2), price)

    def test_predict_initialization_and_model_loaded(self):
        self.assertEqual(self.predictor.model_path, self.model_path)
        self.assertEqual(self.predictor.feature_path, self.feature_path)
        self.assertIsNotNone(self.predictor.model)
        self.assertListEqual(self.predictor.features, self.__class__.features)

    def test_encoding_maps_yes_no_and_m_f_correctly(self):
        df = pd.DataFrame([self.input_dict])
        encoded = self.predictor.encoding(df)

        # YES/NO columns
        self.assertEqual(int(encoded.loc[0, "Claim_3_Years"]), 1)
        self.assertEqual(int(encoded.loc[0, "Owner_Employment_Status"]), 0)
        self.assertEqual(int(encoded.loc[0, "Accidental_Damage"]), 1)
        self.assertEqual(int(encoded.loc[0, "Alarm_Present"]), 0)
        self.assertEqual(int(encoded.loc[0, "Locks_Present"]), 1)
        self.assertEqual(int(encoded.loc[0, "Flooding"]), 0)
        self.assertEqual(int(encoded.loc[0, "Safe_Installed"]), 1)

        # M/F
        self.assertEqual(int(encoded.loc[0, "Owner_Sex"]), 1)

    def test_missing_columns_are_filled_and_prediction_still_works(self):
        # Missing some keys on purpose
        partial_input = {
            "Bedrooms": 2,
            "YearBuilt": 1995,
            "Owner_Sex": "F",
            "Claim_3_Years": "No",
        }

        price = self.predictor.predict_price(partial_input)
        self.assertIsInstance(price, float)
        self.assertEqual(price, self.expected_price)