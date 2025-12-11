# tests/test_result.py

import unittest
import os
import numpy as np

from Health_Insurance.result import result
from Health_Insurance.training import Training, save


class TestResult(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Run once before all tests: train and save a small model."""
        print("Setting up class resources...")
        cls.model_path = "Health_Insurance/test_random_forest.pkl"

        model,rmse,r2 = Training(n=50, split=0.2)
        save(model, cls.model_path)

    @classmethod
    def tearDownClass(cls):
        print("Cleaning up class resources...")
        if os.path.exists(cls.model_path):
            os.remove(cls.model_path)

    def setUp(self):
        print("Setting up before a test...")
        self.age = 30
        self.sex = "male"
        self.bmi = 28.5
        self.children = 1
        self.smoker = "No"
        self.region = "Southwest"

        self.res = result(
            age=self.age,
            sex=self.sex,
            bmi=self.bmi,
            children=self.children,
            smoker=self.smoker,
            region=self.region,
            path=self.model_path,
        )

    def tearDown(self):
        print("Cleaning up after a test...")

    def test_test1(self):
        pred = self.res.predict()

        self.assertIsInstance(pred, np.ndarray)

        self.assertEqual(len(pred), 1)

        self.assertTrue(np.issubdtype(pred.dtype, np.number))

        self.assertGreater(pred[0], 0)

    def test_test2(self):
        self.assertEqual(self.res.age, self.age)

        self.assertEqual(self.res.region, self.region)

        pred1 = self.res.predict()
        pred2 = self.res.predict()
        self.assertEqual(len(pred1), len(pred2))

        self.assertTrue(np.allclose(pred1, pred2))

if __name__ == "__main__":
    unittest.main()
