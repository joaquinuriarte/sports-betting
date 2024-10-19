import pandas as pd
from typing import Tuple
from modules.dataset_generator.configuration_loader import ConfigurationLoader
from modules.dataset_generator.dataset_loader_factory import DatasetLoaderFactory
from modules.dataset_generator.dataset_generator_strategy_factory import DatasetGeneratorStrategyFactory
from modules.dataset_generator.dataset_loader_interface import IDatasetLoader
from modules.dataset_generator.dataset_generator_strategy_interface import IDatasetGeneratorStrategy
from modules.dataset_generator.join_factory import JoinFactory
from modules.dataset_generator.feature_processor_factory import FeatureProcessorFactory

class DatasetGeneration:
    """
    Main orchestrator for dataset generation.
    """

    def __init__(self, config_path: str):
        # Step 1: Load configuration using ConfigurationLoader
        self.config_loader = ConfigurationLoader(config_path)
        self.config = self.config_loader.load_config()

        # Step 2: Instantiate DatasetLoader to load data sources
        self.dataset_loader: IDatasetLoader = DatasetLoaderFactory.create_loader(self.config['sources'])

        # Step 3: Use DatasetGeneratorStrategyFactory to get the appropriate strategy
        # TODO I don't like that we are hard coding this here and we don't have an interface for DatasetGeneratorStrategyFactory. Need to fix this
        join_factory = JoinFactory()
        feature_processor_factory = FeatureProcessorFactory()
        strategy_factory = DatasetGeneratorStrategyFactory(join_factory, feature_processor_factory)
        self.dataset_generation_strategy: IDatasetGeneratorStrategy = strategy_factory.create_generator(
            strategy_name=self.config['strategy'],
            join_operation_type=self.config.get('join_operation', None),
            feature_processing_type=self.config['feature_processing']
        )

    def generate(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate the dataset by loading data, optionally joining, and processing features.
        
        Returns:
            features_df: DataFrame containing the features.
            labels_df: DataFrame containing the labels.
        """
        # Load the data sources
        dataframes = self.dataset_loader.load_data()

        # Generate the features and labels using the strategy
        features_df, labels_df = self.dataset_generation_strategy.generate(dataframes)

        return features_df, labels_df