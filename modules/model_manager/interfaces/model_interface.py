from abc import ABC, abstractmethod
from typing import List, Any, Dict
import pandas as pd
from ...data_structures.model_dataset import ModelDataset, Attribute

class IModel(ABC):
    """
    Interface for any model implementation.
    Defines common operations for setup, training, saving, and inference.
    """

    @abstractmethod
    def forward(self, x: List[List[Attribute]]) -> List[Attribute]:
        """
        Defines the forward pass of the model. Should be implemented in the subclass.
        Args:
            x (List[List[Attribute]]): The input feature set to be processed by the model.
        Returns:
            List[Attribute]: The output after passing through the model's layers.
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
    def train(self, features: List[List[Attribute]], labels: List[List[Attribute]], epochs: int, batch_size: int):
        """
        Trains the model using the provided features and labels.
        Args:
            features (List[List[Attribute]]): The input features for training.
            labels (List[Attribute]): The target labels for training.
            epochs (int): Number of epochs for training.
            batch_size (int): Batch size for training.
        """
        pass

    @abstractmethod
    def predict(self, x: List[List[Attribute]]) -> pd.DataFrame:
        """
        Generates predictions for the provided input data.
        Args:
            x (List[List[Attribute]]): The input feature set for prediction.
        Returns:
            List[Attribute]: The predicted output.
        """
        pass

    @abstractmethod
    def get_training_config(self) -> Dict[str, Any]:
        """
        Gets the current training configuration for the model.

        Returns:
            dict: Dictionary containing the full model configuration.
        """
        pass
