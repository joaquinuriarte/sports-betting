from abc import ABC, abstractmethod
from typing import Optional, List
from modules.data_structures.model_dataset import ModelDataset
from .model_interface import IModel


class ITrainer(ABC):
    """
    Interface for a Trainer implementation.
    Defines the contract for training-related operations.
    """

    @abstractmethod
    def __init__(self, checkpoint_dir: Optional[str] = None) -> None:
        """
        Initializes the Trainer with optional checkpointing.

        Args:
            checkpoint_dir (Optional[str]): Directory to save training checkpoints. If None, checkpoints are not saved.
        """
        pass

    @abstractmethod
    def train(
        self,
        model: IModel,
        train_dataset: ModelDataset,
        val_dataset: Optional[ModelDataset] = None,
    ) -> None:
        """
        Trains the model using the provided dataset.

        Args:
            models (IModel): Models to be trained.
            train_datasets (ModelDataset): Training dataset for the model.
            val_datasets (Optional[ModelDataset]): Validation dataset for each model. If None, no validation is performed.
        """
        pass
