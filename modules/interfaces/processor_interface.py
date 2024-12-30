from abc import ABC, abstractmethod
from typing import Tuple, Optional
from modules.data_structures.model_dataset import ModelDataset


class IProcessor(ABC):
    """
    Interface for processors responsible for generating model datasets.
    """

    @abstractmethod
    def generate(
        self, val_dataset_flag: Optional[bool] = True
    ) -> Tuple[ModelDataset, Optional[ModelDataset]]:
        """
        Generates the training and optional validation datasets.

        Args:
            val_dataset_flag (Optional[bool]): Flag to indicate whether to create a validation dataset.

        Returns:
            Tuple[ModelDataset, Optional[ModelDataset]]: Training dataset and optionally a validation dataset.
        """
        pass
