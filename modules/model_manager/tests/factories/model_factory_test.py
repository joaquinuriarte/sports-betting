import unittest
from unittest.mock import patch, Mock
from ...factories.model_factory import ModelFactory
from ...interfaces.model_interface import IModel
from ...implementations.tensorflow_model import TensorFlowModel


class ModelFactoryTest(unittest.TestCase):

    def setUp(self) -> None:
        """
        Set up a ModelFactory instance for testing.
        """
        self.factory = ModelFactory()

    def test_create_tensorflow_model(self) -> None:
        """
        Test if ModelFactory creates a TensorFlow model when 'tensorflow' type is provided.
        """
        # Mock architecture configuration
        mock_architecture_config = {
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
            ],  # Mock input features
            "output_features": "label",  # Mock output feature
        }

        # Create the TensorFlow model using the factory
        model = self.factory.create("tensorflow", architecture=mock_architecture_config)

        # Assertions
        self.assertIsInstance(model, TensorFlowModel)
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
