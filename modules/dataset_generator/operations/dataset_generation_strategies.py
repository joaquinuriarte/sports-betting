import pandas as pd
from typing import Tuple
from modules.dataset_generator.interfaces.strategy_interface import IDatasetGeneratorStrategy

class JoinBasedGenerator(IDatasetGeneratorStrategy):
    """
    Implements a strategy to join multiple data sources to generate a unified DataFrame.
    """

    def __init__(self, join_operation: object, feature_processor: object): #TODO change obkect for interface
        self.join_operation = join_operation
        self.feature_processor = feature_processor

    def generate(self, dataframes: list) -> Tuple[pd.DataFrame, pd.DataFrame]: #TODO  consider adding data structure interface for this?
        """
        Generate features and labels by joining dataframes and processing features.
        
        Args:
            dataframes (list): List of DataFrames to be joined and processed.
        
        Returns:
            features_df: DataFrame containing the features.
            labels_df: DataFrame containing the labels.
        """
        if self.join_operation:
            joined_df = self.join_operation.perform_join(dataframes)
        else:
            joined_df = dataframes[0]  # If no join is required, use the first dataframe

        features_df, labels_df = self.feature_processor.process(joined_df)
        return features_df, labels_df

class NoJoinGenerator(IDatasetGeneratorStrategy):
    """
    Implements a strategy to generate features and labels without any join operation.
    """

    def __init__(self, feature_processor: object): #TODO change obkect for interface
        self.feature_processor = feature_processor

    def generate(self, dataframes: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Use the first dataframe directly without any join
        features_df, labels_df = self.feature_processor.process(dataframes[0])
        return features_df, labels_df