from abc import ABC, abstractmethod
import pandas as pd

class IModelManager(ABC):
    @abstractmethod
    def train(self, features: pd.DataFrame, labels: pd.DataFrame):
        """Trains the model using features and labels."""
        pass

    @abstractmethod
    def save_model(self, path: str):
        """Saves the model weights to the specified path."""
        pass

    @abstractmethod
    def predict(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """Runs inference on the new data and returns predictions."""
        pass

    