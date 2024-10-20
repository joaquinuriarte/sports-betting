from modules.dataset_generator.operations.dataset_generation_strategies import JoinBasedGenerator, NoJoinGenerator
from modules.dataset_generator.interfaces.strategy_interface import IDatasetGeneratorStrategy
from modules.dataset_generator.interfaces.feature_processor_operator_interface import IFeatureProcessorOperator
from modules.dataset_generator.interfaces.join_operator_interface import IJoinOperator
from typing import List


class StrategyFactory:
    """
    Factory for creating dataset generation strategies based on the configuration.
    """

    @staticmethod
    def create_strategy(strategy_name: str, feature_processor: IFeatureProcessorOperator, join_operations: List[IJoinOperator]) -> IDatasetGeneratorStrategy:
        """
        Creates the appropriate dataset generation strategy.
        
        Args:
            strategy_name (str): Name of the dataset generation strategy (e.g., 'join_based', 'no_join').
            feature_processor (IFeatureProcessorOperator): The feature processor instance used for feature processing.
            join_operations (List[IJoinOperator]): A list of join operation instances to be applied (only applicable if join operations are needed).

        Returns:
            IDatasetGeneratorStrategy: An instance of the appropriate dataset generation strategy.
        """
        if strategy_name == 'join_based':
            dataset_generation_strategy: IDatasetGeneratorStrategy = JoinBasedGenerator(join_operations, feature_processor)
            return dataset_generation_strategy
        elif strategy_name == 'no_join':
            dataset_generation_strategy: IDatasetGeneratorStrategy = NoJoinGenerator(feature_processor)
            return dataset_generation_strategy
        else:
            raise ValueError(f"Unsupported strategy name: {strategy_name}")
