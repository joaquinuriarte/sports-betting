from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd

class IDatasetGeneratorStrategy(ABC):
    """
    Interface for dataset generation strategies.
    """
    @abstractmethod
    def generate(self, dataframes: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass