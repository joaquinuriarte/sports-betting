from abc import ABC, abstractmethod
import pandas as pd
from modules.data_structures.processed_dataset import ProcessedDataset


class IFeatureProcessorOperator(ABC):
    """
    Interface for feature processors.
    """

    @abstractmethod
    def process(self, dataframe: pd.DataFrame) -> ProcessedDataset:
        pass
