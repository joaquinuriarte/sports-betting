import tensorflow as tf
import pandas as pd
from ..interfaces.model_interface import IModel
from typing import Any

class TensorFlowModel(IModel):
    """
    A TensorFlow model wrapper that implements the IModel interface.
    
    Attributes:
        model (tf.keras.Model): The TensorFlow model instance.
    """
    def __init__(self, architecture_config: dict):
        # Store the model configuration
        self.model_config = architecture_config

        # Initialize the model using the architecture configuration
        self.model = self._initialize_model(self.model_config)

    def _initialize_model(self, architecture_config: dict):
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
                    activation=layer_config.get("activation", None)
                )(x)
        outputs = x
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=architecture_config.get("optimizer", "adam"),
            loss=architecture_config.get("loss", "mse"),
            metrics=architecture_config.get("metrics", ["accuracy"])
        )
        return model

    def forward(self, x: Any) -> Any:
        """
        Defines the forward pass of the model.
        
        Args:
            x (Any): Input data, typically a list of lists or similar Python structure.
        
        Returns:
            tf.Tensor: Output after passing through the model's layers.
        """
        # Convert the input to a TensorFlow tensor
        input_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        return self.model(input_tensor)

    def train(self, features: Any, labels: Any, epochs: int, batch_size: int):
        """
        Trains the model using the provided features and labels.
        
        Args:
            features (Any): Input features, typically a list of lists or similar Python structure.
            labels (Any): Target labels, typically a list or similar Python structure.
            epochs (int): Number of epochs to train the model.
            batch_size (int): Batch size to use during training.
        """
        # Convert features and labels to TensorFlow tensors
        features_tensor = tf.convert_to_tensor(features, dtype=tf.float32)
        labels_tensor = tf.convert_to_tensor(labels, dtype=tf.float32)

        # Train the model
        self.model.fit(features_tensor, labels_tensor, epochs=epochs, batch_size=batch_size)

    def predict(self, x: tf.Tensor) -> pd.DataFrame:
        """
        Runs inference on the input tensor and returns predictions.
        
        Args:
            input_tensor (tf.Tensor): New input data for inference.
        
        Returns:
            pd.DataFrame: Predictions for the input data.
        """
        predictions = self.forward(input_tensor)
        return pd.DataFrame(predictions.numpy())

    def save(self, path: str):
        """
        Saves the model weights to the specified path.
        
        Args:
            path (str): Path to save the model weights.
        """
        self.model.save_weights(path)

    def load(self, path: str):
        """
        Loads the model weights from the specified path.
        
        Args:
            path (str): Path from which to load the model weights.
        """
        self.model.load_weights(path)

    def get_training_config(self) -> dict:
        """
        Gets the current training configuration for the model.
        
        Returns:
            dict: Dictionary containing the full model configuration.
        """
        return self.model_config