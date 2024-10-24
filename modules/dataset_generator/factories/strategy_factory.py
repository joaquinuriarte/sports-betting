from modules.dataset_generator.operations.dataset_generation_strategies import (
    JoinBasedGenerator,
    NoJoinGenerator,
)
from modules.dataset_generator.interfaces.strategy_interface import (
    IDatasetGeneratorStrategy,
)
from modules.dataset_generator.interfaces.feature_processor_operator_interface import (
    IFeatureProcessorOperator,
)
from modules.dataset_generator.interfaces.join_operator_interface import IJoinOperator
from modules.dataset_generator.interfaces.factory_interface import IFactory
from modules.data_structures.dataset_config import JoinOperation
from typing import List, Any


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
        if type_name == "join_based":
            dataset_generation_strategy: IDatasetGeneratorStrategy = JoinBasedGenerator(
                join_operations= kwargs.get("join_operations"), feature_processor=kwargs.get("feature_processor")
            )
            return dataset_generation_strategy
        elif type_name == "no_join":
            dataset_generation_strategy: IDatasetGeneratorStrategy = NoJoinGenerator(
                feature_processor=kwargs.get("feature_processor")
            )
            return dataset_generation_strategy
        else:
            raise ValueError(f"Unsupported strategy name: {type_name}")
