import unittest
from Car_Insurance.training import CarInsurance, result
from Car_Insurance.preprocessing import age_factor,mileage_factor,experience_factor,car_age_factor,experience_factor

class TestCarInsurance(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("Starting TestCarInsurance...")

    @classmethod
    def tearDownClass(cls):
        print("Ending TestCarInsurance...")

    def setUp(self):
        self.model = CarInsurance()   # Create fresh object before each test

    def tearDown(self):
        pass  # Nothing to clean up, but required by assignment

    def test_preprocessing_factor(self):
        # Four assertions required per test case
        self.assertIsInstance(age_factor(25), float)
        self.assertEqual(age_factor(999), 1.0)
        self.assertGreater(mileage_factor(15000), 0)
        self.assertLess(experience_factor(0), 2)

    def test_final_premium(self):
        premium = self.model.final_premium(25, 15000, 4, 5, 1, "sedan")
        
        self.assertIsInstance(premium, float)
        self.assertGreater(premium, 0)
        self.assertGreaterEqual(premium, 400)  
        self.assertLess(premium, 2000)

    def test_result_helper(self):
        p1 = result(25, 15000, 4, 5, 1, "sedan")
        p2 = result(60, 5000, 2, 20, 0, "truck")

        self.assertNotEqual(p1, p2)
        self.assertGreater(p1, 0)
        self.assertIsInstance(p1, float)
        self.assertIsInstance(p2, float)


if __name__ == "__main__":
    unittest.main()