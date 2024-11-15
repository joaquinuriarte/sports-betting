import pandas as pd
from typing import List
from modules.data_structures.processed_dataset import ProcessedDataset
from modules.dataset_generator.interfaces.strategy_interface import (
    IDatasetGeneratorStrategy,
)
from ..interfaces.feature_processor_operator_interface import (
    IFeatureProcessorOperator,
)
from main.data_structures.dataset_config import JoinOperation


class JoinBasedGenerator(IDatasetGeneratorStrategy):
    """
    Implements a strategy to join multiple data sources to generate a unified DataFrame.
    """

    def __init__(
        self,
        join_operations: List[JoinOperation],
        feature_processor: IFeatureProcessorOperator,
    ):
        self.join_operations = join_operations
        self.feature_processor = feature_processor

    def generate(self, dataframes: List[pd.DataFrame]) -> ProcessedDataset:
        """
        Generate features and labels by joining dataframes and processing features.

        Args:
            dataframes (List[pd.DataFrame]): List of DataFrames to be joined sequentially.

        Returns:
            ProcessedDataset: The generated dataset containing features and labels.
        """
        result_df = dataframes[0]
        for i, join_info in enumerate(self.join_operations):
            operator = join_info["operator"]
            keys = join_info["keys"]
            right_df = dataframes[i + 1]
            result_df = operator.perform_join(result_df, right_df, keys)

        processed_dataset: ProcessedDataset = self.feature_processor.process(result_df)
        return processed_dataset


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
        processed_dataset: ProcessedDataset = self.feature_processor.process(
            dataframes[0]
        )

        # Return wrapped ProcessedDataset
        return processed_dataset
