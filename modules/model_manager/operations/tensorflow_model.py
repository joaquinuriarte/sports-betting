import tensorflow as tf
import pandas as pd
from ..interfaces.model_interface import IModel
from ...data_structures.processed_dataset import ProcessedDataset

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

    def forward(self, x: tf.Tensor) -> tf.Tensor:
        """
        Defines the forward pass of the model.
        
        Args:
            x: Input tensor.
        
        Returns:
            Output after passing through the model's layers.
        """
        return self.model(x)

    def train(self, processed_dataset: ProcessedDataset):
        epochs = self.model_config.training_epochs
        batch_size = self.model_config.batch_size
        self.model.fit(processed_dataset.features, processed_dataset.labels, epochs=epochs, batch_size=batch_size)
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
        # Convert new_data DataFrame to a tensor before making predictions
        input_tensor = tf.convert_to_tensor(new_data.values, dtype=tf.float32)
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