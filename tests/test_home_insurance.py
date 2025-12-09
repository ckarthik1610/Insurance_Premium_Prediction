import unittest
import os
import pickle
import numpy as np
from sklearn.dummy import DummyRegressor

from Home_Insurance.predict_price import Predict  

class TestHomePredict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("Setting up class resources...")

        cls.model_path = "Home_Insurance/test_home_model.pkl"
        cls.feature_path = "Home_Insurance/test_home_features.pkl"

        cls.features = ["area", "bedrooms", "bathrooms"]

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

        self.input_dict = {
            "area": 1000,
            "bedrooms": 3,
            "bathrooms": 2,
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
