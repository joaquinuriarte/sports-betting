from modules.dataset_generator.operations.dataset_generation_strategies import JoinBasedGenerator, NoJoinGenerator
from modules.dataset_generator.interfaces.dataset_generator_strategy_interface import IDatasetGeneratorStrategy
from modules.dataset_generator.interfaces.join_factory_interface import IJoinFactory
from modules.dataset_generator.interfaces.feature_processor_factory_interface import IFeatureProcessorFactory

class DatasetGeneratorStrategyFactory:
    """
    Factory for creating dataset generation strategies based on the configuration.
    """

    def __init__(self, join_factory: IJoinFactory, feature_processor_factory: IFeatureProcessorFactory):
        self.join_factory = join_factory
        self.feature_processor_factory = feature_processor_factory

    def create_generator(self, strategy_name: str, join_operation_type: str, feature_processing_type: str) -> IDatasetGeneratorStrategy:
        """
        Creates the appropriate dataset generation strategy.
        
        Args:
            strategy_name (str): Name of the dataset generation strategy.
            join_operation_type (str): Type of join operation to use (optional).
            feature_processing_type (str): Type of feature processor to use.
        
        Returns:
            DatasetGeneratorStrategy: An instance of the dataset generation strategy.
        """
        join_operation = None
        if join_operation_type:
            join_operation = self.join_factory.create_join(join_operation_type)
        
        feature_processor = self.feature_processor_factory.create_processor(feature_processing_type)
        
        if strategy_name == 'join_based':
            return JoinBasedGenerator(join_operation, feature_processor)
        # Additional strategies can be added here
        elif strategy_name == 'no_join':
            return NoJoinGenerator(feature_processor)
        else:
            raise ValueError(f"Unsupported strategy name: {strategy_name}")