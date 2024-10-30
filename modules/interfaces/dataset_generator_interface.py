from abc import ABC, abstractmethod
from ..data_structures.processed_dataset import ProcessedDataset

class IDatasetGenerator(ABC):
    """
    Interface for dataset generators.
    """

    @abstractmethod
    def generate(self) -> ProcessedDataset:
        """
        Generates the dataset, including features and labels.

        Returns:
            ProcessedDataset: An object containing the features and labels data.
        """
        pass
