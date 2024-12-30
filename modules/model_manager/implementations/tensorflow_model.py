import tensorflow as tf
import numpy as np
import pandas as pd
from typing import List
from modules.model_manager.interfaces.model_interface import IModel
from modules.data_structures.model_dataset import Example
from modules.data_structures.model_config import ModelConfig


class TensorFlowModel(IModel):
    """
    A TensorFlow model wrapper that implements the IModel interface.

    Attributes:
        model (tf.keras.Model): The TensorFlow model instance.
    """

    def __init__(self, model_config: ModelConfig) -> None:
        # Store the model configuration
        self.model_config = model_config

        # Initialize the model using the architecture configuration
        self.model = self._initialize_model()

        # Store Model variables
        self.output_features: str = self.model_config.architecture["output_features"]

    def _initialize_model(self) -> tf.keras.Model:
        """
        Initializes the model based on the architecture configuration.

        Args:
            architecture_config (dict): Configuration for building the model architecture.

        Returns:
            tf.keras.Model: The initialized TensorFlow model.
        """
        inputs = tf.keras.Input(
            shape=(self.model_config.architecture["input_size"],))
        x = inputs
        for layer_config in self.model_config.architecture["layers"]:
            if layer_config["type"] == "Dense":
                x = tf.keras.layers.Dense(
                    units=layer_config["units"],
                    activation=layer_config.get("activation", None),
                )(x)
        outputs = tf.keras.layers.Dense(units=1)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=self.model_config.architecture.get("optimizer", "adam"),
            # Need to understand why we need to use logits=True
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=self.model_config.architecture.get(
                "metrics", ["accuracy"]),
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
        # Extract features dynamically excluding the output feature
        feature_array = np.array(
            [
                [
                    example.features[feature_name][0] if feature_name in example.features else 0.0
                    for feature_name in example.features
                    if feature_name != self.output_features  # Exclude output feature
                ]
                for example in examples
            ],
            dtype=np.float32,
        )

        # Ensure feature array matches expected input size
        expected_input_size = self.model_config.architecture["input_size"]
        if feature_array.shape[1] != expected_input_size:
            raise ValueError(
                f"Feature array has {feature_array.shape[1]} features, but the model expects {expected_input_size}."
            )

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
        # Extract features dynamically excluding the output feature
        feature_array = np.array(
            [
                [
                    example.features[feature_name][0] if feature_name in example.features else 0.0
                    for feature_name in example.features
                    if feature_name != self.output_features  # Exclude output feature
                ]
                for example in examples
            ],
            dtype=np.float32,
        )

        # Ensure feature array matches expected input size
        expected_input_size = self.model_config.architecture["input_size"]
        if feature_array.shape[1] != expected_input_size:
            raise ValueError(
                f"Feature array has {feature_array.shape[1]} features, but the model expects {expected_input_size}."
            )

        label_array = np.array(
            [
                (
                    example.features[self.output_features][0]
                    if self.output_features in example.features
                    else 0.0
                )
                for example in examples
            ],
            dtype=np.float32,
        )

        # Convert arrays to tensors
        features_tensor = tf.convert_to_tensor(feature_array)
        labels_tensor = tf.convert_to_tensor(label_array)

        # Train the model
        self.model.fit(
            features_tensor, labels_tensor, epochs=epochs, batch_size=batch_size
        )

    def predict(self, examples: List[Example]) -> pd.DataFrame:
        """
        Generates predictions for the provided examples.

        Args:
            examples (List[Example]): A list of `Example` instances.

        Returns:
            pd.DataFrame: The predicted output.
        """
        output_tensor = self.forward(examples)
        predictions = tf.sigmoid(output_tensor).numpy()  # Apply sigmoid
        rounded_predictions = np.round(predictions)     # Apply rounding
        prediction_df = pd.DataFrame(
            rounded_predictions, columns=[
                f"output_{i}" for i in range(rounded_predictions.shape[1])]
        )
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

    def get_training_config(self) -> ModelConfig:
        """
        Gets the current training configuration for the model.

        Returns:
            dict: Dictionary containing the full model configuration.
        """
        return self.model_config
