from ..interfaces.model_manager_interface import IModelManager
from .interfaces.trainer_interface import ITrainer
from .interfaces.predictor_interface import IPredictor
from .helpers.configuration_loader import ConfigurationLoader
from .factories.model_factory import ModelFactory
from .interfaces.model_interface import IModel
from ..data_structures.model_config import ModelConfig
from ..data_structures.model_dataset import ModelDataset, Example
from typing import List, Optional, Tuple
import pandas as pd
import os

class ModelManager(IModelManager):
    """
    Orchestrates model setup, training, saving, and inference.
    """

    def __init__(
        self,
        trainer: ITrainer,
        predictor: IPredictor,
        model_factory: ModelFactory,
        config_loader: ConfigurationLoader,
    ) -> None:
        """
        Initializes the ModelManager with necessary dependencies including trainer, predictor, model factory, and config loader.
        """
        # Instantiate dependencies
        self.trainer = trainer
        self.predictor = predictor
        self.model_factory = model_factory
        self.config_loader = config_loader
    
    def create_models(
        self,
        yaml_path: List[str],
    ) -> List[Tuple[IModel, ModelConfig]]:
        """
        Creates models from a list of YAML configuration paths.

        Args:
            yaml_paths (List[str]): A list of paths to YAML configuration files.

        Returns:
            List[Tuple[IModel, ModelConfig]]: A list of tuples containing model instances and their configurations.
        """
        models_and_config = []

        # Create Models and Model configs
        for yaml in yaml_path:
            model_config: ModelConfig = self.config_loader.load_config(yaml)
            model: IModel = self.model_factory.create(model_config.get("type_name"), model_config)

            models_and_config.append((model, model_config))
        
        return models_and_config
    
    def train(
        self,
        models: List[IModel],
        train_val_datasets: List[Tuple[ModelDataset, ModelDataset]],
        save_after_training: Optional[bool] = True,
    ) -> None:
        """
        Trains the provided models using corresponding training and validation datasets.

        Args:
            models (List[IModel]): The models to be trained.
            train_val_datasets (List[Tuple[ModelDataset, ModelDataset]]): A list of tuples containing training and validation datasets for each model.
            save_after_training (Optional[bool]): Flag to indicate if the model should be saved after training. Default is True.
        """
        # Verify correct input dimensions        
        if len(models) != len(train_val_datasets):
            raise ValueError("Number of models and train_val_datasets provided must be equal.")

        # Train models
        for model, (train_dataset, val_dataset) in zip(models, train_val_datasets):
            self.trainer.train(model, train_dataset, val_dataset)

            # Save model
            if save_after_training:
                self.save(model)

    def predict(
        self,
        models: List[IModel],
        input_data: List[List[Example]],
    ) -> List[pd.DataFrame]:
        """
        Makes predictions using the provided models and input data.

        Args:
            models (List[IModel]): A list of models to be used for prediction.
            input_data (List[List[Example]]): A list of input data examples for each model.

        Returns:
            List[pd.DataFrame]: A list of predictions for each model.
        """
        # Verify correct input dimensions
        if len(models) != len(input_data):
            raise ValueError("Number of models and input_data provided must be equal.")
        
        # Extract and predict
        output_data = []
        for model, examples in zip(models, input_data): 
            output_data.append(self.predictor.predict(model, examples))
        
        return output_data

    def save(
        self,
        model: IModel,
        save_path: Optional[str] = None
    ) -> None:
        """
        Saves the model weights and configuration using the model signature.

        Args:
            model (IModel): The model instance to be saved.
            save_path (Optional[str]): The path to save the model weights. If not provided, default directory is used.
        """
        if not save_path: 
            # Get model signature
            model_signature = model.get_training_config().get("model_signature")

            # Create directory path
            model_directory = os.path.join("models", model_signature)
            os.makedirs(model_directory, exist_ok=True)

            # Create model weights path
            save_path = os.path.join(
                model_directory, f"model_weights_{model_signature}.pth"
            )
        else:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save model weights
        model.save(save_path)
        print(f"Model saved successfully at: {save_path}")

    def load_models(
        self, 
        yaml_paths: List[str], 
        weights_paths: List[str],
    ) -> List[Tuple[IModel, ModelConfig]]:
        """
        Loads models from provided YAML paths and weight paths.

        Args:
            yaml_paths (List[str]): A list of paths to YAML configuration files.
            weights_paths (List[str]): A list of paths to saved model weights.

        Returns:
            List[Tuple[IModel, ModelConfig]]: A list of tuples containing the loaded model instances and their configurations.
        """
        # Verify correct input dimensions
        if len(yaml_paths) != len(weights_paths):
            raise ValueError("Number of yaml_paths and weights_paths provided must be equal.")
        
        models_and_configs = []

        # Loop over yaml and weights and instantiate model and load its weights 
        for yaml, weights_path in zip(yaml_paths, weights_paths): 
            model_config: ModelConfig = self.config_loader.load_config(yaml)
            model: IModel = self.model_factory.create(model_config.get("type_name"), model_config)
            model.load(weights_path)

            models_and_configs.append((model, model_config))
        
        return models_and_configs