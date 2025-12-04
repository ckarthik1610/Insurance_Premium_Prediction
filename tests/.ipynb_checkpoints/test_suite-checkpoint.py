import unittest
from test_car_insurance import TestCarInsurance



def suite():
    suite = unittest.TestSuite()
    
    suite.addTest(unittest.makeSuite(TestCarInsurance))
    suite.addTest(unittest.makeSuite(TestHelper))



if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())