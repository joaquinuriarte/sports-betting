from abc import ABC, abstractmethod
from ...data_structures.model_dataset import Attribute
from typing import List

class IModel(ABC):
    """
    Interface for any model implementation.
    Defines common operations for setup, training, saving, and inference.
    """

    @abstractmethod
    def forward(self, x):
        """
        Defines the forward pass of the model. Should be implemented in the subclass.
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """
        Saves the model weights to the specified path.
        
        Args:
            path (str): The file path where the model weights should be saved.
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """
        Loads the model weights from the specified path.
        
        Args:
            path (str): The file path from which to load the model weights.
        """
        pass

    @abstractmethod
    def train(self, features: List[List[Attribute]], labels: List[Attribute], epochs: int, batch_size: int):
        """
        Trains the model using the provided features and labels.

        Args:
            features (List[List[Attribute]]): The input features for training.
            labels (List[Attribute]): The target labels for training.
        """
        pass
