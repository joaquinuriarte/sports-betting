import pandas as pd
from abc import ABC, abstractmethod

class IJoinOperator(ABC):
    """
    Interface for join operations.
    """
    @abstractmethod
    def perform_join(self, dataframes: list) -> pd.DataFrame:
        pass