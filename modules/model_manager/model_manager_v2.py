# Imports
from ..interfaces.model_manager_interface import IModelManager
from .interfaces.trainer_interface import ITrainer
from .interfaces.predictor_interface import IPredictor
from .helpers.configuration_loader import ConfigurationLoader
from .factories.model_factory import ModelFactory
from .interfaces.model_interface import IModel
from ..data_structures.model_config import ModelConfig
from ..data_structures.model_dataset import ModelDataset, Example
from typing import List, Optional
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

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
        checkpoint_dir: Optional[str] = None,
    ) -> None:
        # Instantiate dependencies
        self.trainer = trainer
        self.predictor = predictor
        self.model_factory = model_factory
        self.config_loader = config_loader
        self.checkpoint_dir = checkpoint_dir
    
    def create_models(
        self,
        yaml_path: List[str],
    ) -> List[tuple(IModel, ModelConfig)]:
        # Initiate final returned list
        models_and_config = []

        # Create Models and Model configs
        for yaml in yaml_path:
            model_config: ModelConfig = self.config_loader.load_config(yaml)
            model: IModel = self.model_factory.create(model_config.get("type_name"), model_config)

            # Append to final list
            models_and_config.append(tuple(model, model_config))
        
        return models_and_config
    
    def train(
        self,
        models: List[IModel],
        train_val_datasets: List[tuple(ModelDataset, ModelDataset)],
    ) -> None:
        # Verify correct input dimensions
        if len(models) != len(train_val_datasets):
            raise ValueError("Number of models and train_val_datasets provided must be equal.")
        # Train models
        for (model, train_dataset) in enumerate(zip(models, train_val_datasets)):
            # Train models
            self.trainer.train(model, train_dataset[0], train_dataset[1])

    def predict(
        self,
        models: List[IModel],
        input_data: List[List[Example]],
    ) -> pd.DataFrame:
        # Verify correct input dimensions
        if len(models) != len(input_data):
            raise ValueError("Number of models and input_data provided must be equal.")
        
        # Extract and predict
        output_data = []
        for (model, input_data) in enumerate(zip(models, input_data)): 
            output_data.append(self.predict(model, input_data))
        
        # Return predicitons
        return output_data

    # def save() -> called automatically after train, ver logic previous save method

    # def load(yaml path, weight path) -> tuple(LoadedModel, ModelConfig)

    # TODO
        #1. Ver si update interface
        #2. Ver si code correct
        #3. 