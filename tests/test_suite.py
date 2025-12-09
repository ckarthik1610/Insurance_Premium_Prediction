
import unittest

from tests.test_car_insurance import TestCarInsurance
from tests.testHelper import TestHelper

from tests.health_suite import health_suite

from tests.test_home_insurance import TestHomePredict
from tests.test_home_insurance2 import TestHomeData

def suite():
    loader = unittest.defaultTestLoader
    test_suite = unittest.TestSuite()
    
    test_suite.addTests(loader.loadTestsFromTestCase(TestCarInsurance))
    test_suite.addTests(loader.loadTestsFromTestCase((TestHelper))
    
    test_suite.addTests(health_suite())
	
    test_suite.addTests(loader.loadTestsFromTestCase((TestHomePredict))
    test_suite.addTests(loader.loadTestsFromTestCase((TestHomeData))

    return test_suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
