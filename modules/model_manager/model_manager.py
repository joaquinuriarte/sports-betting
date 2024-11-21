from ..interfaces.model_manager_interface import IModelManager
from ..data_structures.model_config import ModelConfig
from .interfaces.model_interface import IModel
from .configuration_loader import ConfigurationLoader
from ..data_structures.model_dataset import ModelDataset, Example
from .interfaces.trainer_interface import ITrainer
from .interfaces.predictor_interface import IPredictor
from typing import List
import pandas as pd
import os


class ModelManager(IModelManager):
    """
    Orchestrates model setup, training, saving, and inference.
    """

    def __init__(
        self,
        config_path: List[str],
        model: List[IModel],
        predictor: List[IPredictor],
        trainer: List[ITrainer],
    ):
        # Step 1: Load model configuration 
        # TODO List capability
        self.config_path = config_path
        self.config_loader = ConfigurationLoader(self.config_path)
        self.model_config: ModelConfig = self.config_loader.load_config()

        #Step 2: Load trainers, predictors, models
        #TODO Add list capability
        self.model = model
        self.predictor = predictor
        self.trainer = trainer

        # Store model signature
        # TODO Assess
        self.model_signature = self.model_config.model_signature

        # Step 4: Load existing model weights if specified in the config
        #  TODO Assess
        if self.model_config.model_path:
            self.load_model(self.model_config.model_path)

    # This is supposed to just dispatch Trainer jobs with model and train dataset for each 
    def train(self, model_dataset: ModelDataset, auto_save: bool = True) -> None:
        """
        Trains the model using the provided processed dataset.

        Args:
            model_dataset (ModelDataset): Dataset to train the model with.
        """
        self.trainer.train(self.model, model_dataset)

        # TODO Decide if this is the right approach. Not sure if it reduces flexibility, or increases safety
        if auto_save:
            self.save_model()

    def save_model(self) -> None:
        """
        Saves the model weights and configuration using the model signature.
        """
        model_directory = os.path.join("models", self.model_signature)
        os.makedirs(model_directory, exist_ok=True)

        # Save model weights
        model_weights_path = os.path.join(
            model_directory, f"model_weights_{self.model_signature}.pth"
        )
        self.model.save(model_weights_path)

        # Use ConfigurationLoader to update the model configuration with path
        self.config_loader.update_config(
            self.config_path, "model.save_path", model_weights_path
        )

        # Save the updated YAML configuration alongside the model weights
        config_save_path = os.path.join(
            model_directory, f"model_config_{self.model_signature}.yaml"
        )
        with open(config_save_path, "w") as config_file:
            with open(self.config_path, "r") as original_config:
                config_file.write(original_config.read())

        print(f"Model saved successfully in directory: {model_directory}")

    def load_model(self, path: str) -> None:
        """
        Loads the model weights from the specified path.

        Args:
            path (str): Path from which to load the model weights.
        """
        self.model.load(path)

    def predict(self, prediction_input: List[Example]) -> pd.DataFrame:
        """
        Uses the model to make predictions on new input data.

        Args:
            prediction_input (PredictionInput): New input data for inference.

        Returns:
            pd.DataFrame: Predictions for the input data.
        """
        predictions = self.predictor.predict(self.model, prediction_input)
        return predictions
