import tensorflow as tf
import pandas as pd
from ..interfaces.model_interface import IModel

class TensorFlowModel(IModel):
    """
    A TensorFlow model wrapper that implements the ModelInterface.
    
    Attributes:
        model (tf.keras.Model): The TensorFlow model instance.
    """
    def __init__(self, architecture_config: dict):
        self.model = self._initialize_model(architecture_config)

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

    def train(self, features: pd.DataFrame, labels: pd.DataFrame, epochs: int = 10, batch_size: int = 32):
        """
        Trains the model using the provided features and labels.
        
        Args:
            features (pd.DataFrame): The input features for training.
            labels (pd.DataFrame): The target labels for training.
            epochs (int): Number of epochs to train the model.
            batch_size (int): Batch size to use during training.
        """
        self.model.fit(features, labels, epochs=epochs, batch_size=batch_size)

    def predict(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Runs inference on the new data and returns predictions.
        
        Args:
            new_data (pd.DataFrame): New input data for inference.
        
        Returns:
            pd.DataFrame: Predictions for the input data.
        """
        predictions = self.model.predict(new_data)
        return pd.DataFrame(predictions)

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