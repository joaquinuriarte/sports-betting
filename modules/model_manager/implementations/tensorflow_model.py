import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Any, Dict, List
from ..interfaces.model_interface import IModel
from modules.data_structures.model_dataset import Example

class TensorFlowModel(IModel):
    """
    A TensorFlow model wrapper that implements the IModel interface.

    Attributes:
        model (tf.keras.Model): The TensorFlow model instance.
    """

    def __init__(self, architecture_config: Dict[str, Any]) -> None:
        # Store the model configuration
        self.model_config = architecture_config

        # Initialize the model using the architecture configuration
        self.model = self._initialize_model(self.model_config)

        # Store Model variables
        self.input_features: List[str] = self.model_config["input_features"] 
        self.output_features: str = self.model_config["output_features"]

    def _initialize_model(self, architecture_config: Dict[str, Any]) -> tf.keras.Model:
        """
        Initializes the model based on the architecture configuration.

        Args:
            architecture_config (dict): Configuration for building the model architecture.

        Returns:
            tf.keras.Model: The initialized TensorFlow model.
        """
        inputs = tf.keras.Input(shape=(architecture_config["input_size"],))
        x = inputs
        for layer_config in architecture_config["layers"]:
            if layer_config["type"] == "Dense":
                x = tf.keras.layers.Dense(
                    units=layer_config["units"],
                    activation=layer_config.get("activation", None),
                )(x)
        outputs = x
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=architecture_config.get("optimizer", "adam"),
            loss=architecture_config.get("loss", "mse"),
            metrics=architecture_config.get("metrics", ["accuracy"]),
        )
        return model

    def forward(self, examples: List[Example]) -> tf.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            examples (List[Example]): A list of `Example` instances.

        Returns:
            tf.Tensor: Output after passing through the model's layers.
        """
        # Convert examples to TensorFlow tensor
        feature_array = np.array([
            [
                feature[input_feature] 
                for input_feature in self.input_features
                for feature in example.features 
                if input_feature in feature
            ]
            for example in examples
        ], dtype=np.float32)
        features_tensor = tf.convert_to_tensor(feature_array)

        return self.model(features_tensor)

    def train(self, examples: List[Example], epochs: int, batch_size: int) -> None:
        """
        Trains the model using the provided examples.

        Args:
            examples (List[Example]): A list of `Example` instances containing features and labels.
            epochs (int): Number of epochs to train the model.
            batch_size (int): Batch size to use during training.
        """
        # Extract features and labels from examples
        feature_array = np.array([
            [
                next(feature[input_feature] for feature in example.features if input_feature in feature)
                for input_feature in self.input_features
            ]
            for example in examples
        ], dtype=np.float32)

        label_array = np.array([
            next(feature[self.output_features] for feature in example.features if self.output_features in feature)
            for example in examples
        ], dtype=np.float32)

        # Convert arrays to tensors
        features_tensor = tf.convert_to_tensor(feature_array)
        labels_tensor = tf.convert_to_tensor(label_array)

        # Train the model
        self.model.fit(features_tensor, labels_tensor, epochs=epochs, batch_size=batch_size)


    def predict(self, examples: List[Example]) -> pd.DataFrame:
        """
        Generates predictions for the provided examples.

        Args:
            examples (List[Example]): A list of `Example` instances.

        Returns:
            pd.DataFrame: The predicted output.
        """
        output_tensor = self.forward(examples)
        predictions = output_tensor.numpy()
        prediction_df = pd.DataFrame(predictions, columns=[f'output_{i}' for i in range(predictions.shape[1])])
        return prediction_df

    def save(self, path: str) -> None:
        """
        Saves the model weights to the specified path.

        Args:
            path (str): Path to save the model weights.
        """
        self.model.save_weights(path)

    def load(self, path: str) -> None:
        """
        Loads the model weights from the specified path.

        Args:
            path (str): Path from which to load the model weights.
        """
        self.model.load_weights(path)

    def get_training_config(self) -> Dict[str, Any]:
        """
        Gets the current training configuration for the model.

        Returns:
            dict: Dictionary containing the full model configuration.
        """
        return self.model_config
