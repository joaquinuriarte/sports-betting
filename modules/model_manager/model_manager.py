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
        # Instantiate dependencies
        self.trainer = trainer
        self.predictor = predictor
        self.model_factory = model_factory
        self.config_loader = config_loader
    
    def create_models(
        self,
        yaml_path: List[str],
    ) -> List[Tuple(IModel, ModelConfig)]:
        # Initiate final returned list
        models_and_config = []

        # Create Models and Model configs
        for yaml in yaml_path:
            model_config: ModelConfig = self.config_loader.load_config(yaml)
            model: IModel = self.model_factory.create(model_config.get("type_name"), model_config)

            # Append to final list
            models_and_config.append((model, model_config))
        
        return models_and_config
    
    def train(
        self,
        models: List[IModel],
        train_val_datasets: List[Tuple(ModelDataset, ModelDataset)],
        save_after_training: Optional[bool] = True,
    ) -> None:
        # Verify correct input dimensions
        if len(models) != len(train_val_datasets):
            raise ValueError("Number of models and train_val_datasets provided must be equal.")

        # Train models
        for model, (train_dataset, val_dataset) in zip(models, train_val_datasets):
            # Train models
            self.trainer.train(model, train_dataset, val_dataset)

            # Save model
            if save_after_training:
                self.save(model)

    def predict(
        self,
        models: List[IModel],
        input_data: List[List[Example]],
    ) -> List[pd.DataFrame]:
        # Verify correct input dimensions
        if len(models) != len(input_data):
            raise ValueError("Number of models and input_data provided must be equal.")
        
        # Extract and predict
        output_data = []
        for model, examples in zip(models, input_data): 
            output_data.append(self.predict(model, examples))
        
        # Return predicitons
        return output_data

    def save(
        self,
        model: IModel,
        save_path: Optional[str] = None
    ) -> None:
        """
        Saves the model weights and configuration using the model signature.
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

        # Save model weights
        model.save(save_path)

        print(f"Model saved successfully in directory: {model_directory}")

    def load_models(
        self, 
        yaml_paths: List[str], 
        weights_paths: List[str],
    ) -> List[Tuple(IModel, ModelConfig)]:
        """
        Loads a model from yaml path with weights and configuration.
        """
        # Verify correct input dimensions
        if len(yaml_paths) != len(weights_paths):
            raise ValueError("Number of yaml_paths and weights_paths provided must be equal.")
        
        # Create empty return object
        models_and_configs = []

        # Loop over yaml and weights and instantiate model and load its weights 
        for yaml, weights_path in zip(yaml_paths, weights_paths): 
            model_config: ModelConfig = self.config_loader.load_config(yaml)
            model: IModel = self.model_factory.create(model_config.get("type_name"), model_config)
            model.load(weights_path)

            models_and_configs.append((model, model_config))
        
        # Return final list
        return models_and_configs


    # TODO
        #1. Ver si update interface
            # YA
        #2. Ver si code correct
        #3. add comments y method descriptions
        #4. Signatures link yaml y weights, pichea save path. Quita de yaml, ModelConfig, and configLoader
            # YA

    # Model manager test cases and all sub components
    # black and mypy for all
    # dataset gen ver si bring back factories and config
    # Main meterle 