from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd

class DataIO(ABC):

    @abstractmethod
    def read_df_from_path(self, model_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Reads and optionally joins data based on the model configuration.

        Parameters:
        - model_config (Dict[str, Any]): Configuration dictionary including paths, columns, and optional joins.

        Returns:
        - pd.DataFrame: The resulting dataframe after loading and optional joining operations.
        """
        pass