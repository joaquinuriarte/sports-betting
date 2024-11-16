from abc import ABC, abstractmethod
import pandas as pd
from ..data_structures.model_dataset import ModelDataset, Example
from typing import List

class IModelManager(ABC):
    @abstractmethod
    def train(self, model_dataset: ModelDataset, auto_save: bool) -> None:
        """Trains the model using features and labels."""
        pass

    @abstractmethod
    def save_model(self) -> None:
        """Saves the model weights to the specified path."""
        pass

    @abstractmethod
    def predict(self, prediction_input: List[Example]) -> pd.DataFrame:
        """Runs inference on the new data and returns predictions."""
        pass

    @abstractmethod
    def load_model(self, path: str) -> None:
        """Loads the model weights to the model."""
        pass
