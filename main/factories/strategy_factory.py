from ..implementations.dataset_generation_strategies import (
    JoinBasedGenerator,
    NoJoinGenerator,
)
from modules.dataset_generator.interfaces.strategy_interface import (
    IDatasetGeneratorStrategy,
)
from ..interfaces.feature_processor_operator_interface import (
    IFeatureProcessorOperator,
)
from ...modules.interfaces.factory_interface import IFactory
from main.data_structures.dataset_config import JoinOperation
from typing import Any, List, cast


class StrategyFactory(IFactory[IDatasetGeneratorStrategy]):
    """
    Factory for creating dataset generation strategies based on the configuration.
    """

    @staticmethod
    def create(type_name: str, *args: Any, **kwargs: Any) -> IDatasetGeneratorStrategy:
        """
        Creates the appropriate dataset generation strategy.

        Args:
            strategy_name (str): Name of the dataset generation strategy (e.g., 'join_based', 'no_join').
            feature_processor (IFeatureProcessorOperator): The feature processor instance used for feature processing.
            join_operations (List[IJoinOperator]): A list of join operation instances to be applied (only applicable if join operations are needed).

        Returns:
            IDatasetGeneratorStrategy: An instance of the appropriate dataset generation strategy.
        """
        feature_processor = cast(
            IFeatureProcessorOperator, kwargs.get("feature_processor")
        )
        join_operations = cast(List[JoinOperation], kwargs.get("join_operations", []))

        if type_name == "join_based":
            return JoinBasedGenerator(
                join_operations=join_operations, feature_processor=feature_processor
            )
        elif type_name == "no_join":
            return NoJoinGenerator(feature_processor=feature_processor)
        else:
            raise ValueError(f"Unsupported strategy name: {type_name}")
