from modules.dataset_generator.interfaces.strategy_interface import IDatasetGeneratorStrategy
from modules.dataset_generator.factories.feature_processor_factory import FeatureProcessorFactory
from modules.dataset_generator.factories.join_factory import JoinFactory
from modules.dataset_generator.factories.strategy_factory import StrategyFactory
from modules.dataset_generator.interfaces.feature_processor_operator_interface import IFeatureProcessorOperator
from modules.dataset_generator.interfaces.join_operator_interface import IJoinOperator

class DatasetStrategyCreator:
    """
    Creates dataset generation strategies based on the configuration.

    This class takes in configuration details, such as the strategy name, join operation type, and feature processing type,
    and utilizes the join and feature processor factories to create the appropriate dataset generation strategy.
    """

    def __init__(self, strategy_name: str, join_operation_type: str, feature_processing_type: str):
        self.strategy_name = strategy_name
        self.join_operation_type = join_operation_type
        self.feature_processing_type = feature_processing_type

    def create_strategy(self) -> IDatasetGeneratorStrategy:
        """
        Creates the appropriate dataset generation strategy.
        
        Args:
            strategy_name (str): Name of the dataset generation strategy (e.g., 'join_based', 'no_join').
            join_operation_type (str): Type of join operation to use (optional, e.g., 'inner', 'left', 'right').
            feature_processing_type (str): Type of feature processor to use (e.g., 'top_n_players').

        Returns:
            IDatasetGeneratorStrategy: An instance of the dataset generation strategy.
        """
        # Create feature processor instance
        feature_processor_factory: FeatureProcessorFactory = FeatureProcessorFactory()
        feature_processor: IFeatureProcessorOperator = feature_processor_factory.create_processor(self.feature_processing_type)

        # Create join operations list if join operation type is specified
        join_operations = []
        join_factory: JoinFactory = JoinFactory()
        if self.join_operation_type:
            for _ in range(len(self.join_operation_type.split(','))):
                join_operator: IJoinOperator = join_factory.create_join(self.join_operation_type)
                join_operations.append(join_operator)

        # Create and return strategy
        strategy_factory: StrategyFactory = StrategyFactory()
        dataset_generation_strategy: IDatasetGeneratorStrategy = strategy_factory.create_strategy(self.strategy_name, feature_processor, join_operations)

        # Return instantiated strategy
        return dataset_generation_strategy