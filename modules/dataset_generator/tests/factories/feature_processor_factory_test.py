import unittest
from typing import cast
from modules.dataset_generator.factories.feature_processor_factory import (
    FeatureProcessorFactory,
)
from modules.dataset_generator.implementations.feature_processing_operations_v0 import (
    TopNPlayersFeatureProcessorV0,
)


class TestFeatureProcessorFactory(unittest.TestCase):
    """
    Unit tests for the FeatureProcessorFactory class.
    """

    def test_create_top_n_players_processor(self) -> None:
        """
        Test that the factory creates a TopNPlayersFeatureProcessor instance when 'top_n_players' type is provided.
        """
        processor = FeatureProcessorFactory.create(
            "top_n_players",
            top_n_players=5,
            sorting_criteria="PTS",
            player_stats_columns=["PTS", "AST"],
        )
        processor = cast(TopNPlayersFeatureProcessorV0, processor)
        self.assertIsInstance(processor, TopNPlayersFeatureProcessorV0)
        self.assertEqual(processor.top_n_players, 5)
        self.assertEqual(processor.sorting_criteria, "PTS")
        self.assertEqual(processor.player_stats_columns, ["PTS", "AST"])

    def test_create_top_n_players_processor_defaults(self) -> None:
        """
        Test that the factory creates a TopNPlayersFeatureProcessor instance with default values if no kwargs are provided.
        """
        processor = FeatureProcessorFactory.create("top_n_players")
        processor = cast(TopNPlayersFeatureProcessorV0, processor)
        self.assertIsInstance(processor, TopNPlayersFeatureProcessorV0)
        self.assertEqual(processor.top_n_players, 8)  # Default value
        self.assertEqual(processor.sorting_criteria, "MIN")  # Default value
        self.assertEqual(processor.player_stats_columns,
                         ["MIN"])  # Default value

    def test_create_unsupported_type(self) -> None:
        """
        Test that the factory raises a ValueError when an unsupported processing type is provided.
        """
        with self.assertRaises(ValueError) as context:
            FeatureProcessorFactory.create("invalid_type")
        self.assertEqual(
            str(context.exception), "Unsupported feature processing type: invalid_type"
        )


if __name__ == "__main__":
    unittest.main()
