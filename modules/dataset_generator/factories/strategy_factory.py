from modules.dataset_generator.operations.dataset_generation_strategies import JoinBasedGenerator, NoJoinGenerator
from modules.dataset_generator.interfaces.strategy_interface import IDatasetGeneratorStrategy
from typing import List


class StrategyFactory:
    """
    Factory for creating dataset generation strategies based on the configuration.
    """

    @staticmethod
    def create_strategy(strategy_name: str, feature_processor: object, join_operations: List[object]) -> IDatasetGeneratorStrategy: #TODO change object for interface
        """
        Creates the appropriate dataset generation strategy.
        
        Args:
            # TODO Fix
        
        Returns:
            DatasetGeneratorStrategy: An instance of the dataset generation strategy.
        """
        
        if strategy_name == 'join_based':
            return JoinBasedGenerator(join_operations, feature_processor)
        # Additional strategies can be added here
        elif strategy_name == 'no_join':
            return NoJoinGenerator(feature_processor)
        else:
            raise ValueError(f"Unsupported strategy name: {strategy_name}")
