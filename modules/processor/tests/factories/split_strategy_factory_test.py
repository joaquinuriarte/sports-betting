import unittest
from modules.processor.implementations.random_split_strategy import RandomSplitStrategy
from modules.processor.factories.split_strategy_factory import SplitStrategyFactory
from modules.processor.interfaces.split_strategy_interface import ISplitStrategy


class TestSplitStrategyFactory(unittest.TestCase):
    """
    Test cases for the SplitStrategyFactory class.
    """

    def setUp(self):
        """
        Set up the factory instance for tests.
        """
        self.factory = SplitStrategyFactory()

    def test_create_random_split_strategy(self):
        """
        Test creating a RandomSplitStrategy instance.
        """
        strategy = self.factory.create("random_split")
        self.assertIsInstance(strategy, RandomSplitStrategy)
        self.assertIsInstance(strategy, ISplitStrategy)

    def test_create_invalid_strategy(self):
        """
        Test creating an invalid strategy type.
        """
        with self.assertRaises(ValueError) as context:
            self.factory.create("invalid_strategy")
        
        self.assertEqual(
            str(context.exception), "Unsupported split strategy type: invalid_strategy"
        )


if __name__ == "__main__":
    unittest.main()
