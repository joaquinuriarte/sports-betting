from modules.interfaces.factory_interface import IFactory
from modules.model_manager.interfaces.model_interface import IModel
from modules.model_manager.implementations.tensorflow_model_v0 import TensorFlowModelV0
from modules.model_manager.implementations.tensorflow_model_v01 import TensorFlowModelV01

from typing import Any


class ModelFactory(IFactory[IModel]):
    """
    Factory for creating models based on the configuration.
    """

    def create(self, type_name: str, *args: Any, **kwargs: Any) -> IModel:
        """
        Creates an instance of a model based on the provided type name and configuration.

        Args:
            type_name (str): The name/type of the model to create (e.g., "tensorflow").
            architecture_config (dict): The architecture configuration for the model.

        Returns:
            IModel: An instance of the model.
        """
        if type_name == "tensorflow_model_v0":
            # Retrieve the model_config from args if present
            model_config = kwargs.get("model_config")
            if model_config is None:
                raise ValueError(
                    "model_config must be provided to create a model")
            return TensorFlowModelV0(model_config)
        elif type_name == "tensorflow_model_v01":
            # Retrieve the model_config from args if present
            model_config = kwargs.get("model_config")
            if model_config is None:
                raise ValueError(
                    "model_config must be provided to create a model")
            return TensorFlowModelV01(model_config)
        else:
            raise ValueError(f"Unsupported model type: {type_name}")
