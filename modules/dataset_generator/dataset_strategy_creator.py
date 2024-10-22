from modules.dataset_generator.interfaces.strategy_interface import (
    IDatasetGeneratorStrategy,
)
from modules.dataset_generator.interfaces.feature_processor_operator_interface import (
    IFeatureProcessorOperator,
)
from modules.dataset_generator.interfaces.factory_interface import IFactory
from modules.data_structures.dataset_config import DatasetConfig


class DatasetStrategyCreator:
    """
    Creates dataset generation strategies based on the configuration.
    """

    def __init__(
        self,
        config: DatasetConfig,
        feature_processor_factory: IFactory,
        join_factory: IFactory,
        strategy_factory: IFactory,
    ):
        """
        Initializes the strategy creator with the provided configuration and factories.

        Args:
            config (DatasetConfig): The dataset configuration containing strategy details.
            feature_processor_factory (IFactory): Factory to create feature processors.
            join_factory (IFactory): Factory to create join operations.
            strategy_factory (IFactory): Factory to create strategies.
        """
        self.config = config
        self.feature_processor_factory = feature_processor_factory
        self.join_factory = join_factory
        self.strategy_factory = strategy_factory

    def create_strategy(self) -> IDatasetGeneratorStrategy:
        """
        Creates the appropriate dataset generation strategy.

        Returns:
            IDatasetGeneratorStrategy: An instance of the dataset generation strategy.
        """
        # Step 1: Create feature processor instance
        feature_processor: IFeatureProcessorOperator = (
            self.feature_processor_factory.create(self.config.feature_processor)
        )

        # Create join operations with keys
        join_operations = [
            {"operator": self.join_factory.create(join), "keys": join["keys"]}
            for join in self.config.joins
        ]

        # Step 3: Create and return the strategy using the strategy factory
        dataset_generation_strategy: IDatasetGeneratorStrategy = (
            self.strategy_factory.create(
                self.config.strategy, feature_processor, join_operations
            )
        )

        return dataset_generation_strategy
