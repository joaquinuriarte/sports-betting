from .helpers.dataset_loader import DatasetLoader
from .interfaces.strategy_interface import IDatasetGeneratorStrategy
from modules.data_structures.processed_dataset import ProcessedDataset
from modules.data_structures.dataset_config import DatasetConfig
from ..interfaces.dataset_generator_interface import IDatasetGenerator
from .helpers.configuration_loader import ConfigurationLoader
from .interfaces.feature_processor_operator_interface import (
    IFeatureProcessorOperator,
)
from modules.interfaces.factory_interface import IFactory
from .interfaces.join_operator_interface import IJoinOperator
from ..data_structures.dataset_config import JoinOperation
from .interfaces.data_io_interface import DataIO
from typing import List

class DatasetGenerator(IDatasetGenerator):
    """
    Main orchestrator for dataset generation.
    """

    def __init__(
        self,
        yaml_path: str,
        configuration_loader: ConfigurationLoader,
        data_factory: IFactory[DataIO],
        feature_processor_factory: IFactory[IFeatureProcessorOperator],
        join_factory: IFactory[IJoinOperator],
        strategy_factory: IFactory[IDatasetGeneratorStrategy],
    ):
        # Instantiate dependencies 
        self.configuration_loader = configuration_loader
        self.data_factory = data_factory
        self.join_factory = join_factory
        self.feature_processor_factory = feature_processor_factory
        self.strategy_factory = strategy_factory

        # create datasetConfig con loader
        dataset_config: DatasetConfig = self.configuration_loader.load_config(yaml_path)

        # create loader (list of data readers)
        self.dataset_loader: DatasetLoader = self.create_loader(dataset_config, self.data_factory)

        # create strategy
        self.dataset_strategy: IDatasetGeneratorStrategy = self.create_strategy(dataset_config, self.feature_processor_factory, self.join_factory, self.strategy_factory)

    def generate(self) -> ProcessedDataset:
        """
        Generate the dataset by loading data, optionally joining, and processing features.

        Returns:
            ProcessedDataset: The generated dataset containing features and labels.
        """
        # Load the data sources
        dataframes = self.dataset_loader.load_data()

        # Generate the features and labels using the strategy
        processed_dataset: ProcessedDataset = self.dataset_strategy.generate(dataframes)

        return processed_dataset
    
    def create_loader(self, config: DatasetConfig, factory: IFactory[DataIO]) -> DatasetLoader:
        """
        Creates a DatasetLoader object with the appropriate data readers.

        Returns:
            DatasetLoader: A DatasetLoader instance initialized with the correct data readers.
        """
        data_loaders = []

        # Iterate over each Source object in the configuration
        for source in config.sources:
            # Use the factory to create a DataIO reader based on the file type
            data_io: DataIO = factory.create(source.file_type)

            data_loaders.append(data_io)

        # Create and return the DatasetLoader with the sources and their data readers
        return DatasetLoader(data_loaders, config.sources)

    def create_strategy(
        self,
        config: DatasetConfig,
        feature_processor_factory: IFactory[IFeatureProcessorOperator],
        join_factory: IFactory[IJoinOperator],
        strategy_factory: IFactory[IDatasetGeneratorStrategy],
    ) -> IDatasetGeneratorStrategy:
        """
        Creates the appropriate dataset generation strategy.

        Returns:
            IDatasetGeneratorStrategy: An instance of the dataset generation strategy.
        """
        # Step 1: Create feature processor instance
        feature_processor: IFeatureProcessorOperator = (
            feature_processor_factory.create(
                config.feature_processor_type,
                config.top_n_players,
                config.sorting_criteria,
                config.player_stats_columns,
            )
        )

        # Create join operations with keys
        join_operations: List[JoinOperation] = [
            JoinOperation(
                operator= join_factory.create(join["type"]),
                keys=join["keys"],
            )
            for join in config.joins
        ]

        # Step 3: Create and return the strategy using the strategy factory
        dataset_generation_strategy: IDatasetGeneratorStrategy = (
            strategy_factory.create(
                config.strategy, feature_processor, join_operations
            )
        )

        return dataset_generation_strategy

    
    
