import unittest
import os
import sys
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from Health_Insurance.preprocessing import preprocess
from Health_Insurance.training import Training, save


class TestTraining(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("Setting up class resources...")

        cls.csv_path = "Health_Insurance/insurance_data.csv"
        cls.model_path = "Health_Insurance/test_model.pkl"

    @classmethod
    def tearDownClass(cls):
        print("Cleaning up class resources...")
        if os.path.exists(cls.model_path):
            os.remove(cls.model_path)

    def setUp(self):
        print("Setting up before test...")
        self.prep = preprocess(self.csv_path)

    def tearDown(self):
        print("Cleaning up after test...")

    def test_training_output(self):

        model,rmse,r2 = Training(n=50, split=0.2)    # small forest for fast test   

        self.assertIsInstance(model, Pipeline)

        self.assertIsInstance(model.named_steps["rf"], RandomForestRegressor)

        self.prep.train_test(0.2)
        preds = model.predict(self.prep.X_Test)
        self.assertEqual(len(preds), len(self.prep.X_Test))

        self.assertTrue(np.issubdtype(preds.dtype, np.number))

    def test_save_model(self):

        model,rmse,r2 = Training(n=10, split=0.2)

        save(model, self.model_path)

        self.assertTrue(os.path.exists(self.model_path))

        loaded = joblib.load(self.model_path)

        self.assertIsInstance(loaded, Pipeline)

        self.assertIsInstance(loaded.named_steps["rf"], RandomForestRegressor)

        self.prep.train_test(0.2)
        preds = loaded.predict(self.prep.X_Test)
        self.assertEqual(len(preds), len(self.prep.X_Test))
