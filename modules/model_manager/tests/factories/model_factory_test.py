import unittest
from unittest.mock import patch
from modules.model_manager.factories.model_factory import ModelFactory
from modules.model_manager.interfaces.model_interface import IModel
from modules.model_manager.implementations.tensorflow_model_v10 import TensorFlowModelV10


class ModelFactoryTest(unittest.TestCase):

    def setUp(self) -> None:
        """
        Set up a ModelFactory instance for testing.
        """
        self.factory = ModelFactory()

    @patch("modules.data_structures.model_config.ModelConfig")
    def test_create_tensorflow_model(self, MockModelConfig) -> None:
        """
        Test if ModelFactory creates a TensorFlow model when 'tensorflow' type is provided.
        """
        # Mock ModelConfig
        mock_model_config = MockModelConfig()
        mock_model_config.type_name = "tensorflow"
        mock_model_config.architecture = {
            "input_size": 10,
            "layers": [
                {"type": "Dense", "units": 64, "activation": "relu"},
                {"type": "Dense", "units": 1, "activation": "sigmoid"},
            ],
            "optimizer": "adam",
            "loss": "mse",
            "metrics": ["accuracy"],
            "input_features": [
                "feature1",
                "feature2",
                "feature3",
            ],
            "output_features": "label",
        }

        # Set return value for __getitem__ to return the architecture config when ["architecture"] is accessed
        mock_model_config.__getitem__.side_effect = lambda key: (
            mock_model_config.architecture if key == "architecture" else None
        )

        # Create the TensorFlow model using the factory with the mocked ModelConfig
        model = self.factory.create(
            type_name="tensorflow", model_config=mock_model_config
        )

        # Assertions
        self.assertIsInstance(model, TensorFlowModelV10)
        self.assertIsInstance(model, IModel)

    def test_create_unsupported_model_type(self) -> None:
        """
        Test if ModelFactory raises a ValueError for an unsupported model type.
        """
        with self.assertRaises(ValueError) as context:
            self.factory.create("unsupported_type")

        self.assertEqual(
            str(context.exception), "Unsupported model type: unsupported_type"
        )


if __name__ == "__main__":
    unittest.main()
