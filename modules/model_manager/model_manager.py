from ..interfaces.factory_interface import IFactory
from ..interfaces.model_manager_interface import IModelManager
from ..data_structures.model_config import ModelConfig
from .interfaces.model_interface import IModel
from .configuration_loader import ConfigurationLoader
from ..data_structures.model_dataset import ModelDataset
from ..data_structures.prediction_input import PredictionInput
from .trainer.trainer import Trainer
from .predictor.predictor import Predictor
import pandas as pd
import os

class ModelManager(IModelManager):
    """
    Orchestrates model setup, training, saving, and inference.
    """

    def __init__(self, config_path: str, model_factory: IFactory[IModel]):
        # Step 1: Load model configuration
        self.config_path = config_path
        self.config_loader = ConfigurationLoader(self.config_path)
        self.model_config: ModelConfig = self.config_loader.load_config()

        # Step 2: Instantiate Model using ModelFactory
        self.model: IModel = model_factory.create(
            self.model_config.type_name, 
            self.model_config.architecture
        )

        # Store model signature
        self.model_signature = self.model_config.model_signature

        # Step 3: Load existing model weights if specified in the config
        if self.model_config.model_path:
            self.load_model(self.model_config.model_path)
    
    def train(self, model_dataset: ModelDataset, auto_save: bool = True):
        """
        Trains the model using the provided processed dataset.

        Args:
            model_dataset (ModelDataset): Dataset to train the model with.
        """
        trainer = Trainer()
        trainer.train(self.model, model_dataset)

        # TODO Decide if this is the right approach. Not sure if it reduces flexibility, or increases safety
        if auto_save:
            self.save_model()

    def save_model(self):
        """
        Saves the model weights and configuration using the model signature.
        """
        model_directory = os.path.join("models", self.model_signature)
        os.makedirs(model_directory, exist_ok=True)

        # Save model weights
        model_weights_path = os.path.join(model_directory, f"model_weights_{self.model_signature}.pth")
        self.model.save(model_weights_path)

        # Use ConfigurationLoader to update the model configuration with path
        self.config_loader.update_config(self.config_path, "model.save_path", model_weights_path)

        # Save the updated YAML configuration alongside the model weights
        config_save_path = os.path.join(model_directory, f"model_config_{self.model_signature}.yaml")
        with open(config_save_path, "w") as config_file:
            with open(self.config_path, "r") as original_config:
                config_file.write(original_config.read())

        print(f"Model saved successfully in directory: {model_directory}")

    def load_model(self, path: str):
        """
        Loads the model weights from the specified path.
        
        Args:
            path (str): Path from which to load the model weights.
        """
        self.model.load(path)

    def predict(self, prediction_input: PredictionInput) -> pd.DataFrame:
        """
        Uses the model to make predictions on new input data.

        Args:
            prediction_input (PredictionInput): New input data for inference.
        
        Returns:
            pd.DataFrame: Predictions for the input data.
        """
        predictor = Predictor()
        predictions = predictor.predict(self.model, prediction_input)
        return predictions