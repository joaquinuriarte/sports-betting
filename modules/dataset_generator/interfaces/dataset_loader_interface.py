from abc import ABC, abstractmethod
from typing import List
import pandas as pd


class IDatasetLoader(ABC):
    """
    Interface for dataset loaders.
    Defines common operations for loading data sources as DataFrames.
    """

    @abstractmethod
    def load_data(self) -> List[pd.DataFrame]:
        """
        Loads the data from all sources.

        Returns:
            List[pd.DataFrame]: List of DataFrames for each data source.
        """
        pass
