from abc import ABC, abstractmethod
import pandas as pd
from ..data_structures.model_dataset import ModelDataset, Example
from ..model_manager.interfaces.model_interface import IModel
from ..data_structures.model_config import ModelConfig
from typing import List, Tuple, Optional

class IModelManager(ABC):
    @abstractmethod
    def create_models(
        self,
        yaml_path: List[str],
    ) -> List[Tuple[IModel, ModelConfig]]:
        """Instantiates models from a yaml path and returns models with their configuration object."""
        pass

    @abstractmethod
    def train(
        self,
        models: List[IModel],
        train_val_datasets: List[Tuple[ModelDataset, ModelDataset]],
        save_after_training: Optional[bool] = True,
    ) -> None:
        """Trains the models using their provided ModelDataset."""
        pass

    @abstractmethod
    def predict(
        self,
        models: List[IModel],
        input_data: List[List[Example]],
    ) -> List[pd.DataFrame]:
        """Runs inference on the provided Example List."""
        pass

    @abstractmethod
    def save(
        self,
        model: IModel,
        save_path: Optional[str] = None
    ) -> None:
        """
        Saves the model using the model signature.
        """
        pass

    @abstractmethod
    def load_models(
        self, 
        yaml_paths: List[str], 
        weights_paths: List[str],
    ) -> List[Tuple[IModel, ModelConfig]]:
        """Loads the model weights to the model."""
        pass
