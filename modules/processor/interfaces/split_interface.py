from abc import ABC, abstractmethod
from typing import Tuple
from modules.data_structures.model_dataset import ModelDataset


class ISplitStrategy(ABC):
    """
    Interface for all dataset split strategies.
    """

    @abstractmethod
    def split(self, dataset: ModelDataset, train_percentage: float) -> Tuple[ModelDataset, ModelDataset]:
        """
        Splits the dataset into training and validation datasets.

        Args:
            dataset (ModelDataset): The dataset to split.
            train_percentage (float): The percentage of data to allocate to the training set (e.g., 80 for 80%).

        Returns:
            Tuple[ModelDataset, ModelDataset]: The training and validation datasets.
        """
        pass
