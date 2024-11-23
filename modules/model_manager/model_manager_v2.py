# Imports
from ..interfaces.model_manager_interface import IModelManager
from .interfaces.trainer_interface import ITrainer
from .interfaces.predictor_interface import IPredictor
from .helpers.configuration_loader import ConfigurationLoader
from .factories.model_factory import ModelFactory
from .interfaces.model_interface import IModel
from typing import List

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
    )
