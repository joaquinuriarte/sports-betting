import pandas as pd
from typing import Tuple
from modules.dataset_generator.configuration_loader import ConfigurationLoader
from modules.dataset_generator.dataset_loader_creator import DatasetLoaderCreator
from modules.dataset_generator.interfaces.dataset_loader_interface import IDatasetLoader
from modules.dataset_generator.interfaces.strategy_interface import IDatasetGeneratorStrategy
from modules.dataset_generator.dataset_strategy_creator import DatasetStrategyCreator

class DatasetGeneration:
    """
    Main orchestrator for dataset generation.
    """

    def __init__(self, config_path: str):
        # Step 1: Load configuration using ConfigurationLoader TODO: Do we need to decouple ConfigurationLoader?
        self.config_loader = ConfigurationLoader(config_path)
        self.config = self.config_loader.load_config()

        # Step 2: Instantiate DatasetLoader to load data sources TODO: Do we need to decouple DatasetLoaderCreator?
        self.dataset_loader: IDatasetLoader = DatasetLoaderCreator.create_loader(self.config['sources'])

        # Step 3: Use StrategyFactory to create dataset generation strategy TODO: Do we need to decouple DatasetStrategyCreator?
        self.dataset_strategy: IDatasetGeneratorStrategy = DatasetStrategyCreator(self.config['strategy'], self.config.get('join_operation', None), feature_processing_type=self.config['feature_processing'])

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
        features_df, labels_df = self.dataset_strategy.generate(dataframes)

        return features_df, labels_df