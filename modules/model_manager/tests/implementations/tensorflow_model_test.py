import unittest
from unittest.mock import patch, MagicMock
import tensorflow as tf
import numpy as np
import pandas as pd
from modules.model_manager.implementations.tensorflow_model_v10 import TensorFlowModelV10
from modules.data_structures.model_dataset import Example
from modules.data_structures.model_config import ModelConfig
from typing import List


class TensorFlowModelTest(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up common dependencies for tests.
        """
        # Initialize the model configuration
        self.model_config = ModelConfig(
            model_signature="",
            type_name="tensorflow",
            architecture={
                "input_size": 3,
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
            },
            training={
                "epochs": 20,
                "learning_rate": 0.001,
                "optimizer": "Adam",
                "loss_function": "MSELoss",
                "split_strategy": "random_split",
                "batch_size": 32,
            },
        )

        # Instantiate TensorFlowModel with the sample model configuration
        self.model: TensorFlowModelV10 = TensorFlowModelV10(self.model_config)

        self.examples: List[Example] = [
            Example(
                features={
                    "feature1": [1.0],
                    "feature2": [2.0],
                    "feature3": [3.0],
                    "label": [1.0],
                }
            ),
            Example(
                features={
                    "feature1": [4.0],
                    "feature2": [5.0],
                    "feature3": [6.0],
                    "label": [0.0],
                }
            ),
        ]

    @patch("tensorflow.keras.Model.fit")
    def test_train(self, mock_fit: MagicMock) -> None:
        """
        Test the train method of TensorFlowModel.
        """
        epochs: int = 10
        batch_size: int = 2

        # Call the train method
        self.model.train(self.examples, epochs, batch_size)

        # Assertions to validate training call
        mock_fit.assert_called_once()
        args, kwargs = mock_fit.call_args

        # Extract feature array and label array from args to validate input structure
        features_tensor: tf.Tensor = args[
            0
        ]  # The first positional argument should be features tensor
        labels_tensor: tf.Tensor = args[
            1
        ]  # The second positional argument should be labels tensor

        # Expected feature and label values
        expected_features = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
        )  # Adjusted to 3 features

        expected_labels = np.array([1.0, 0.0], dtype=np.float32)

        # Validate feature and label tensors
        np.testing.assert_array_almost_equal(
            features_tensor.numpy(), expected_features)
        np.testing.assert_array_almost_equal(
            labels_tensor.numpy(), expected_labels)

        # Validate additional arguments such as epochs and batch size
        self.assertEqual(kwargs["epochs"], epochs)
        self.assertEqual(kwargs["batch_size"], batch_size)

    @patch.object(tf.keras.Model, "__call__", autospec=True)
    def test_forward(self, mock_call: MagicMock) -> None:
        """
        Test the forward method of TensorFlowModel.
        """
        # Mock model.__call__ to return a predefined value
        expected_output = tf.constant([[0.5], [0.8]], dtype=tf.float32)
        mock_call.return_value = expected_output

        # Call forward
        output: tf.Tensor = self.model.forward(self.examples)

        # Assertions
        np.testing.assert_array_almost_equal(
            output.numpy(), expected_output.numpy())

    @patch.object(
        TensorFlowModelV10,
        "forward",
        return_value=tf.constant([[0.5], [0.8]], dtype=tf.float32),
    )
    def test_predict(self, mock_forward: MagicMock) -> None:
        """
        Test the predict method of TensorFlowModel.
        """
        # Call predict
        predictions: pd.DataFrame = self.model.predict(self.examples)

        # Assertions
        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertEqual(predictions.shape, (2, 1))
        np.testing.assert_array_almost_equal(
            predictions.values, mock_forward.return_value.numpy()
        )

    @patch("tensorflow.keras.Model.save_weights")
    def test_save(self, mock_save_weights: MagicMock) -> None:
        """
        Test the save method of TensorFlowModel.
        """
        path: str = "path/to/save/weights"
        self.model.save(path)
        mock_save_weights.assert_called_once_with(path)

    @patch("tensorflow.keras.Model.load_weights")
    def test_load(self, mock_load_weights: MagicMock) -> None:
        """
        Test the load method of TensorFlowModel.
        """
        path: str = "path/to/load/weights"
        self.model.load(path)
        mock_load_weights.assert_called_once_with(path)

    def test_get_training_config(self) -> None:
        """
        Test the get_training_config method of TensorFlowModel.
        """
        config = self.model.get_training_config()
        self.assertEqual(config, self.model_config)


if __name__ == "__main__":
    unittest.main()
