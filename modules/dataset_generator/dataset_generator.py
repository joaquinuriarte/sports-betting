import pandas as pd
from typing import Tuple
from modules.dataset_generator.interfaces.dataset_loader_interface import IDatasetLoader
from modules.dataset_generator.interfaces.strategy_interface import IDatasetGeneratorStrategy

class DatasetGeneration:
    """
    Main orchestrator for dataset generation.
    """

    def __init__(self, config_path: str, configuration_loader: IConfigurationLoader, dataset_loader_creator: IDatasetLoaderCreator, dataset_strategy_creator: IDatasetStrategyCreator): # TODO: Pass interfaces instead
        # Step 1: Load configuration using ConfigurationLoader 
        self.config = configuration_loader.load_config(config_path)

        # Step 2: Instantiate DatasetLoader using DatasetLoaderCreator to load data sources
        self.dataset_loader: IDatasetLoader = dataset_loader_creator.create_loader(self.config['sources'])

        # Step 3: Use StrategyFactory to create dataset generation strategy TODO: Do we need to decouple DatasetStrategyCreator?
        self.dataset_strategy: IDatasetGeneratorStrategy = dataset_strategy_creator.create_strategy(self.config['strategy'], self.config.get('join_operation', None), feature_processing_type=self.config['feature_processing'])

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