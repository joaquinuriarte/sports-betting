from ..interfaces.factory_interface import IFactory
from ..interfaces.model_manager_interface import IModelManager
from ..data_structures.model_config import ModelConfig
from .interfaces.model_interface import IModel
from .configuration_loader import ConfigurationLoader
from ..data_structures.model_dataset import ModelDataset
import pandas as pd


class ModelManager(IModelManager):
    """
    Orchestrates model setup, training, saving, and inference.
    """

    def __init__(self, config_path: str, model_factory: IFactory[IModel]):
        # Step 1: Load model configuration
        self.config_loader = ConfigurationLoader(config_path)
        self.model_config: ModelConfig = self.config_loader.load_config()

        # Step 2: Instantiate Model using ModelFactory
        self.model: IModel = model_factory.create(
            self.model_config.type_name, 
            self.model_config.architecture
        )

        # Step 3: Load existing model weights if specified in the config
        if self.model_config.model_path:
            self.load_model(self.model_config.model_path)
    
    def train(self, model_dataset: ModelDataset):
        """
        Trains the model using the provided processed dataset.

        Args:
            model_dataset (ModelDataset): Dataset to train the model with.
        """
        trainer = Trainer()
        trainer.train(self.model, model_dataset)

    def save_model(self, path: str):
        """
        Saves the model weights to the specified path.
        
        Args:
            path (str): Path to save the model weights.
        """
        self.model.save(path)

    def load_model(self, path: str):
        """
        Loads the model weights from the specified path.
        
        Args:
            path (str): Path from which to load the model weights.
        """
        self.model.load(path)

    def predict(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Uses the model to make predictions on new input data.
        
        Args:
            new_data (pd.DataFrame): New input data for inference.
        
        Returns:
            pd.DataFrame: Predictions for the input data.
        """
        predictor = Predictor()
        predictions = predictor.predict(self.model, new_data)
        return predictions