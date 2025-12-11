import unittest
import os
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

from Home_Insurance.Risk_factor import data 
class TestHomeData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        
        cls.csv_path = "Home_Insurance/test_home_data.csv"
        sample_df = pd.DataFrame({
            "Area": [1200, 1500, 1800, 2000],
            "Bedrooms": [2, 3, 4, 5],
            "Age_of_House": [10, 5, 20, 15],
            "Annual_Premium_Price": [1000, 1200, 1600, 1800]
        })

        sample_df.to_csv(cls.csv_path, index=False)

        cls.model_file = "Home_Insurance/Linear_Regression.pkl"
        cls.feature_file = "Home_Insurance/Feature_names.pkl"

    @classmethod
    def tearDownClass(cls):
  
        if os.path.exists(cls.csv_path):
            os.remove(cls.csv_path)

        if os.path.exists(cls.model_file):
            os.remove(cls.model_file)

        if os.path.exists(cls.feature_file):
            os.remove(cls.feature_file)

    def setUp(self):
        print("Setting up before test...")
        self.obj = data(self.csv_path)

    def tearDown(self):
        """Runs after each test."""
        print("Cleaning up after test...")

    def test_preprocess_outputs(self):
        """Test preprocessing logic."""
        self.obj.preprocess()
        self.assertIsInstance(self.obj.data, pd.DataFrame)
        self.assertTrue(all(self.obj.X.dtypes.apply(lambda x: x.kind in "if")))
        self.assertEqual(len(self.obj.X), len(self.obj.response))
        self.assertListEqual(list(self.obj.features), list(self.obj.X.columns))

    def test_train_and_save(self):

        self.obj.preprocess()
        self.obj.train()
        self.obj.save()

        self.assertIsInstance(self.obj.model, LinearRegression)
        self.assertTrue(os.path.exists(self.__class__.model_file))
        self.assertTrue(os.path.exists(self.__class__.feature_file))
        with open(self.__class__.feature_file, "rb") as f:
            saved_features = pickle.load(f)
        self.assertListEqual(list(saved_features), list(self.obj.features))

