from ...interfaces.factory_interface import IFactory
from ..interfaces.model_interface import IModel
from ..operations.tensor_flow_model import TensorFlowModel
from ...data_structures.model_config import ModelConfig
from typing import Any, Dict


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
        if type_name == "tensorflow":
            architecture: Dict[str, Any] = kwargs.get("architecture")
            return TensorFlowModel(architecture)
        else:
            raise ValueError(f"Unsupported model type: {type_name}")
