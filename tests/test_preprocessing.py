
import unittest
import os
import numpy as np
from sklearn.compose import ColumnTransformer
from Health_Insurance.preprocessing import preprocess


class TestPreprocess(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("Setting up class resources...")
        cls.csv_path = r"Health_Insurance/insurance_data.csv"
        cls.preprocessor_obj = preprocess(cls.csv_path)

    @classmethod
    def tearDownClass(cls):
        print("Cleaning up class resources...")

    def setUp(self):
        print("Setting up before a test...")
        self.pp = preprocess(self.csv_path)

    def tearDown(self):
        print("Cleaning up after a test...")

    def test_train_test_split_shapes_and_columns(self):
        self.pp.train_test(split=0.2)
        self.assertEqual(self.pp.X.shape[1], 6)
        self.assertEqual(
            len(self.pp.X_Train) + len(self.pp.X_Test),
            len(self.pp.X)
        )
        self.assertEqual(
            len(self.pp.Y_Train) + len(self.pp.Y_Test),
            len(self.pp.Y)
        )
        self.assertListEqual(
            list(self.pp.X.columns),
            ["age", "sex", "bmi", "children", "smoker", "region"]
        )

    def test_preprocessor_fit_transform_output(self):

        self.pp.train_test(split=0.2)
        preprocessor = self.pp.preprocessing()
        transformed = preprocessor.fit_transform(self.pp.X_Train)
        self.assertEqual(
            transformed.shape[0],
            self.pp.X_Train.shape[0]
        )
        self.assertIsInstance(transformed, np.ndarray)
        self.assertGreater(
            transformed.shape[1],
            self.pp.X_Train.shape[1]
        )
        numeric_slice = transformed[:, -3:]  
        self.assertTrue(numeric_slice.dtype.kind in ["i", "f"])  # int or float

