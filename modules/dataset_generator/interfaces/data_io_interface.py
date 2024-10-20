from abc import ABC, abstractmethod
from typing import List
import pandas as pd

class DataIO(ABC):

    @abstractmethod
    def read_df_from_path(self, path: str, columns: List[str]) -> pd.DataFrame:
        """
        Reads and optionally joins data based on the model configuration.

        Parameters:
        - path (str): Path to data source
        - columns (List[str]): Target columns from data source

        Returns:
        - pd.DataFrame: The resulting dataframe after loading and optional joining operations.
        """
        pass
