from abc import ABC, abstractmethod
import pandas as pd

class IModelManager(ABC):
    @abstractmethod
    def setup_model(self, config: str):
        """Loads model into self.model using the given configuration."""
        pass

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

    