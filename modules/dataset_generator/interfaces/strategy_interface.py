from abc import ABC, abstractmethod
from typing import List
import pandas as pd
from modules.data_structures.processed_dataset import ProcessedDataset


class IDatasetGeneratorStrategy(ABC):
    """
    Interface for dataset generation strategies.
    """

    @abstractmethod
    def generate(self, dataframes: List[pd.DataFrame]) -> ProcessedDataset:
        """
        Generate the dataset by processing the input dataframes.

        Args:
            dataframes (List[pd.DataFrame]): List of input DataFrames.

        Returns:
            ProcessedDataset: The generated dataset containing features and labels.
        """
        pass
