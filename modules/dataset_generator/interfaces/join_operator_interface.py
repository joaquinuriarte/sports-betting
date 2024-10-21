import pandas as pd
from abc import ABC, abstractmethod
from typing import List

class IJoinOperator(ABC):
    """
    Interface for join operations.
    """
    @abstractmethod
    def perform_join(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        pass