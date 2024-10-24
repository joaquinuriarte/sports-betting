from modules.dataset_generator.configuration_loader import ConfigurationLoader
from modules.dataset_generator.dataset_loader_creator import DatasetLoaderCreator
from modules.dataset_generator.dataset_loader import DatasetLoader
from modules.dataset_generator.dataset_strategy_creator import DatasetStrategyCreator
from modules.dataset_generator.interfaces.strategy_interface import (
    IDatasetGeneratorStrategy,
)
from modules.dataset_generator.interfaces.factory_interface import IFactory
from modules.dataset_generator.interfaces.join_operator_interface import IJoinOperator
from modules.dataset_generator.interfaces.feature_processor_operator_interface import (
    IFeatureProcessorOperator,
)
from modules.dataset_generator.interfaces.data_io_interface import DataIO
from modules.data_structures.dataset_config import DatasetConfig
from modules.data_structures.processed_dataset import ProcessedDataset
from ..interfaces.dataset_generator_interface import IDatasetGenerator


class DatasetGenerator(IDatasetGenerator):
    """
    Main orchestrator for dataset generation.
    """

    def __init__(
        self,
        config_path: str,
        data_io_factory: IFactory[DataIO],
        feature_processor_factory: IFactory[IFeatureProcessorOperator],
        join_factory: IFactory[IJoinOperator],
        strategy_factory: IFactory[IDatasetGeneratorStrategy],
    ):
        # Step 1: Load configuration using ConfigurationLoader
        config_loader = ConfigurationLoader(config_path)
        self.config: DatasetConfig = config_loader.load_config()

        # Step 2: Instantiate DatasetLoader through the DatasetLoaderCreator to load data sources
        dataset_loader_creator = DatasetLoaderCreator(self.config, data_io_factory)
        self.dataset_loader: DatasetLoader = dataset_loader_creator.create_loader()

        #  Step 3: Use StrategyFactory to create dataset generation strategy
        dataset_strategy_creator = DatasetStrategyCreator(
            self.config, feature_processor_factory, join_factory, strategy_factory
        )
        self.dataset_strategy: IDatasetGeneratorStrategy = (
            dataset_strategy_creator.create_strategy()
        )

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
