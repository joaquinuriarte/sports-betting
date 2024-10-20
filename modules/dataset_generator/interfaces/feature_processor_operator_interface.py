from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd

class IFeatureProcessorOperator(ABC):
    """
    Interface for feature processors.
    """
    @abstractmethod
    def process(self, dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass