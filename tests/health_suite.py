import unittest

from .test_preprocessing import TestPreprocess
from .test_health_training import TestTraining
from .test_health_result import TestResult


def health_suite():
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestPreprocess))
    suite.addTests(loader.loadTestsFromTestCase(TestTraining))
    suite.addTests(loader.loadTestsFromTestCase(TestResult))

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(health_suite())