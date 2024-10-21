import pandas as pd
from typing import List
from modules.data_structures.processed_dataset import ProcessedDataset
from modules.dataset_generator.interfaces.strategy_interface import IDatasetGeneratorStrategy
from modules.dataset_generator.interfaces.join_operator_interface import IJoinOperator
from modules.dataset_generator.interfaces.feature_processor_operator_interface import IFeatureProcessorOperator

class JoinBasedGenerator(IDatasetGeneratorStrategy):
    """
    Implements a strategy to join multiple data sources to generate a unified DataFrame.
    """

    def __init__(self, join_operations: List[IJoinOperator], feature_processor: IFeatureProcessorOperator):
        """
        Initializes the generator with a join operation and feature processor.

        Args:
            join_operation (IJoinOperator): The operation to perform the join.
            feature_processor (IFeatureProcessorOperator): Processor for feature extraction.
        """
        self.join_operations = join_operations
        self.feature_processor = feature_processor

    def generate(self, dataframes: List[pd.DataFrame]) -> ProcessedDataset:
        """
        Generate features and labels by joining dataframes and processing features.

        Args:
            dataframes (List[pd.DataFrame]): List of DataFrames to be joined and processed.

        Returns:
            ProcessedDataset: The generated dataset containing features and labels.
        """
        # Perform each join operation sequentially
        result_df = dataframes[0]
        for i, join_op in enumerate(self.join_operations):
            result_df = join_op.perform_join([result_df, dataframes[i + 1]])

        # Process the joined dataframe into features and labels
        features_df, labels_df = self.feature_processor.process(result_df)

        # Return wrapped ProcessedDataset
        return ProcessedDataset(features=features_df, labels=labels_df)

class NoJoinGenerator(IDatasetGeneratorStrategy):
    """
    Implements a strategy to generate features and labels without any join operation.
    """

    def __init__(self, feature_processor: IFeatureProcessorOperator):
        """
        Initializes the generator with a feature processor.

        Args:
            feature_processor (IFeatureProcessorOperator): Processor for feature extraction.
        """
        self.feature_processor = feature_processor

    def generate(self, dataframes: List[pd.DataFrame]) -> ProcessedDataset:
        """
        Generate features and labels from the first dataframe without joining.

        Args:
            dataframes (List[pd.DataFrame]): List of DataFrames to be processed.

        Returns:
            ProcessedDataset: The generated dataset containing features and labels.
        """
        # Process the first dataframe directly
        features_df, labels_df = self.feature_processor.process(dataframes[0])

        # Return wrapped ProcessedDataset
        return ProcessedDataset(features=features_df, labels=labels_df)
