
import unittest

from .test_car_insurance import TestCarInsurance
from .testHelper import TestHelper

from .health_suite import health_suite

from .test_home_insurance import TestHomePredict
from .test_home_insurance2 import TestHomeData

def suite():
    loader = unittest.defaultTestLoader
    test_suite = unittest.TestSuite()
    
    test_suite.addTests(loader.loadTestsFromTestCase(TestCarInsurance))
    test_suite.addTests(loader.loadTestsFromTestCase(TestHelper))
    
    test_suite.addTests(health_suite())
	
    test_suite.addTests(loader.loadTestsFromTestCase(TestHomePredict))
    test_suite.addTests(loader.loadTestsFromTestCase(TestHomeData))

    return test_suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
