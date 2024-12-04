from modules.dataset_generator.implementations.feature_processing_operations import (
    TopNPlayersFeatureProcessor,
)
from modules.dataset_generator.interfaces.feature_processor_operator_interface import (
    IFeatureProcessorOperator,
)
from modules.interfaces.factory_interface import IFactory
from typing import Any

class FeatureProcessorFactory(IFactory[IFeatureProcessorOperator]):
    """
    Factory for creating feature processors based on the type specified.
    """

    @staticmethod
    def create(type_name: str, *args: Any, **kwargs: Any) -> IFeatureProcessorOperator:
        """
        Creates a feature processor instance based on the provided processing type.

        Args:
            processing_type (str): The type of feature processing to perform (e.g., 'top_n_players').
            top_n_players (int): Number of top players to include for each team.
            sorting_criteria (str): Statistic used to rank players (e.g., 'MIN' for minutes played).
            player_stats_columns (List[str]): List of player statistics to include in the feature vector.

        Returns:
            IFeatureProcessorOperator: An instance of the appropriate feature processor.

        Raises:
            ValueError: If the provided processing type is not supported.
        """
        # Default values should indicate error with yaml
        if type_name == "top_n_players":
            feature_processor: IFeatureProcessorOperator = TopNPlayersFeatureProcessor(
                top_n_players=kwargs.get("top_n_players", 8),
                sorting_criteria=kwargs.get("sorting_criteria", "MIN"),
                look_back_window=kwargs.get("look_back_window", 10),
                player_stats_columns=kwargs.get("player_stats_columns", ["MIN"]),
            )
            return feature_processor
        else:
            raise ValueError(f"Unsupported feature processing type: {type_name}")
