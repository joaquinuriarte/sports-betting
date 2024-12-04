import unittest
from modules.dataset_generator.factories.strategy_factory import StrategyFactory
from modules.dataset_generator.implementations.dataset_generation_strategies import (
    JoinBasedGenerator,
    NoJoinGenerator,
)
from modules.dataset_generator.interfaces.feature_processor_operator_interface import (
    IFeatureProcessorOperator,
)
from modules.data_structures.dataset_config import JoinOperation
from unittest.mock import Mock


class TestStrategyFactory(unittest.TestCase):
    """
    Unit tests for the StrategyFactory class.
    """

    def setUp(self) -> None:
        """
        Sets up mocks for dependencies before each test.
        """
        # Create mock instances for feature processor and join operations
        self.feature_processor = Mock(spec=IFeatureProcessorOperator)
        self.join_operations = [Mock(spec=JoinOperation) for _ in range(2)]

    def test_create_join_based_strategy(self) -> None:
        """
        Test that the factory creates a JoinBasedGenerator instance when 'join_based' type is provided.
        """
        strategy = StrategyFactory.create(
            "join_based",
            feature_processor=self.feature_processor,
            join_operations=self.join_operations,
        )
        self.assertIsInstance(strategy, JoinBasedGenerator)

    def test_create_no_join_strategy(self) -> None:
        """
        Test that the factory creates a NoJoinGenerator instance when 'no_join' type is provided.
        """
        strategy = StrategyFactory.create(
            "no_join",
            feature_processor=self.feature_processor,
        )
        self.assertIsInstance(strategy, NoJoinGenerator)

    def test_create_unsupported_strategy(self) -> None:
        """
        Test that the factory raises a ValueError when an unsupported strategy type is provided.
        """
        with self.assertRaises(ValueError) as context:
            StrategyFactory.create(
                "unsupported_strategy",
                feature_processor=self.feature_processor,
            )
        self.assertEqual(str(context.exception), "Unsupported strategy name: unsupported_strategy")


if __name__ == "__main__":
    unittest.main()
