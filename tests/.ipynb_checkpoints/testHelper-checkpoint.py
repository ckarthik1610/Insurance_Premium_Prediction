import unittest
import pandas as pd


class TestHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\nStarted TestHelper.")

    @classmethod
    def tearDownClass(cls):
        print("Finished TestHelper.\n")

    def setUp(self):
        self.df = pd.DataFrame({"a": [1, 2, 3]})

    def tearDown(self):
        pass

    def test_dataframe(self):
        self.assertEqual(len(self.df), 3)
        self.assertIn("a", self.df.columns)
        self.assertEqual(self.df["a"].sum(), 6)
        self.assertIsInstance(self.df.iloc[0]["a"], int)

    def test_math(self):
        x, y = 4, 5
        self.assertEqual(x + y, 9)
        self.assertNotEqual(x * y, 10)
        self.assertGreater(y, x)
        self.assertIsInstance(x, int)


if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)