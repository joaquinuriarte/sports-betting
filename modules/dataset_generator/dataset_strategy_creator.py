from modules.dataset_generator.interfaces.strategy_interface import IDatasetGeneratorStrategy
from modules.dataset_generator.factories.feature_processor_factory import FeatureProcessorFactory
from modules.dataset_generator.factories.join_factory import JoinFactory
from modules.dataset_generator.factories.strategy_factory import StrategyFactory
from modules.dataset_generator.interfaces.join_factory_interface import IJoinFactory
from modules.dataset_generator.interfaces.feature_processor_factory_interface import IFeatureProcessorFactory

class DatasetStrategyCreator:
    """
    Creates dataset generation strategies based on the configuration. #TODO: Fix comments
    """

    def __init__(self, strategy_name: str, join_operation_type: str, feature_processing_type: str):
        self.strategy_name = strategy_name
        self.join_operation_type = join_operation_type
        self.feature_processing_type = feature_processing_type

    def create_strategy(self) -> IDatasetGeneratorStrategy:
        """
        Creates the appropriate dataset generation strategy.
        
        Returns:
            DatasetGeneratorStrategy: An instance of the dataset generation strategy.
        """
        # Create feature processor instance #TODO: Decouple from here? & do interface
        feature_processor: IFeatureProcessorFactory = FeatureProcessorFactory(self.feature_processing_type)

        # Create join operations list if join operation type is specified #TODO: Decouple from here? & do interface
        join_operations = []
        join_factory: IJoinFactory = JoinFactory()
        if self.join_operation_type:
            for _ in range(len(self.join_operation_type.split(','))):
                join_operations.append(join_factory.create_join(self.join_operation_type))

        # Create and return strategy
        dataset_generation_strategy: IDatasetGeneratorStrategy = StrategyFactory(self.strategy_name, feature_processor, join_operations)

        # Return instantiated strategy
        return dataset_generation_strategy