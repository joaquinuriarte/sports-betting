from typing import List
import pandas as pd
from abc import ABC, abstractmethod

class IDatasetLoader(ABC):
    """
    Abstract base class for dataset loader.
    """

    @abstractmethod
    def load_data(self) -> List[pd.DataFrame]:
        pass