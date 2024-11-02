from abc import ABC, abstractmethod
import pandas as pd
from ..data_structures.model_dataset import ModelDataset
from ..data_structures.prediction_input import PredictionInput

class IModelManager(ABC):
    @abstractmethod
    def train(self, model_dataset: ModelDataset, auto_save: bool):
        """Trains the model using features and labels."""
        pass

    @abstractmethod
    def save_model(self, path: str):
        """Saves the model weights to the specified path."""
        pass

    @abstractmethod
    def predict(self, prediction_input: PredictionInput) -> pd.DataFrame:
        """Runs inference on the new data and returns predictions."""
        pass

    @abstractmethod
    def load_model(self, path: str): 
        """Loads the model weights to the model."""
        pass
    