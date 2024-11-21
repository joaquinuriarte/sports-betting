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
        models: List[IModel],
        train_datasets: List[ModelDataset],
        val_datasets: Optional[List[ModelDataset]] = None
    ) -> None:
        """
        Trains the models using the provided datasets.

        Args:
            models (List[IModel]): A list of models to be trained.
            train_datasets (List[ModelDataset]): A list of training datasets for each model.
            val_datasets (Optional[List[ModelDataset]]): A list of validation datasets for each model. If None, no validation is performed.
        """
        pass
