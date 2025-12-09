import unittest
from test_car_insurance import TestCarInsurance
from testHelper import TestHelper


def suite():
    test_suite = unittest.TestSuite()
    
    test_suite.addTest(unittest.makeSuite(TestCarInsurance))
    test_suite.addTest(unittest.makeSuite(TestHelper))
    
    return test_suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())