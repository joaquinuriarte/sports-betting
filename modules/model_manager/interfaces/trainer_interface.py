from abc import ABC, abstractmethod
from typing import Optional
from modules.data_structures.model_dataset import ModelDataset
from .model_interface import IModel


class ITrainer(ABC):
    """
    Interface for a Trainer implementation.
    Defines the contract for training-related operations.
    """

    @abstractmethod
    def __init__(self, checkpoint_dir: Optional[str] = None):
        """
        Initializes the Trainer with optional checkpointing.

        Args:
            checkpoint_dir (Optional[str]): Directory to save training checkpoints. If None, checkpoints are not saved.
        """
        pass

    @abstractmethod
    def train(self, model: IModel, model_dataset: ModelDataset):
        """
        Trains the model using the provided dataset.

        Args:
            model (IModel): The model to be trained.
            model_dataset (ModelDataset): The dataset containing features and labels.
        """
        pass
