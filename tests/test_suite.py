
import unittest

from .test_car_insurance import TestCarInsurance
from .testHelper import TestHelper

from .test_preprocessing import TestPreprocess
from .test_health_training import TestTraining
from .test_health_result import TestResult


from .test_home_insurance import TestHomePredict
from .test_home_insurance2 import TestHomeData

def suite():
    loader = unittest.defaultTestLoader
    test_suite = unittest.TestSuite()
    
    test_suite.addTests(loader.loadTestsFromTestCase(TestCarInsurance))
    test_suite.addTests(loader.loadTestsFromTestCase(TestHelper))
    
    test_suite.addTests(loader.loadTestsFromTestCase(TestPreprocess))
    test_suite.addTests(loader.loadTestsFromTestCase(TestTraining))
    test_suite.addTests(loader.loadTestsFromTestCase(TestResult))
	
    test_suite.addTests(loader.loadTestsFromTestCase(TestHomePredict))
    test_suite.addTests(loader.loadTestsFromTestCase(TestHomeData))

    return test_suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
