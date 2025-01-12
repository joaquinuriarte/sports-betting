from abc import ABC, abstractmethod
from modules.data_structures.model_dataset import (
    Example,
)
from typing import Any, List, Optional
import pandas as pd
from modules.data_structures.model_config import ModelConfig


class IModel(ABC):
    """
    Interface for any model implementation.
    Defines common operations for setup, training, saving, and inference.
    """

    @abstractmethod
    def forward(self, examples: List[Example]) -> Any:
        """
        Defines the forward pass of the model. Should be implemented in the subclass.
        Args:
            examples (list): List of Example instances to be processed by the model.
        Returns:
            Any: The output after passing through the model's layers.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Saves the model weights to the specified path."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Loads the model weights from the specified path."""
        pass

    @abstractmethod
    def train(
        self,
        training_examples: List[Example],
        epochs: int,
        batch_size: int,
        validation_examples: Optional[List[Example]] = None,
    ) -> None:
        """
        Trains the model using the provided examples.
        Args:
            examples (list): A list of `Example` objects with features and labels.
            epochs (int): Number of epochs for training.
            batch_size (int): Batch size for training.
        """
        pass

    @abstractmethod
    def predict(
        self, examples: List[Example], return_target_labels: Optional[bool] = False
    ) -> pd.DataFrame:
        """Generates predictions for the provided examples."""
        pass

    @abstractmethod
    def get_training_config(self) -> ModelConfig:
        """Gets the current training configuration for the model."""
        pass
