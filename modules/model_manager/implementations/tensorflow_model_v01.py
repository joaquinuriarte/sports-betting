import tensorflow as tf
import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from keras.callbacks import History
from numpy.typing import NDArray
import datetime
import os
from modules.model_manager.interfaces.model_interface import IModel
from modules.data_structures.model_dataset import Example
from modules.data_structures.model_config import ModelConfig
from modules.model_manager.implementations.model_utils.custom_loss_fxns import mse_plus_hinge_margin_loss


class TensorFlowModelV01(IModel):
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
        self._log_dir_override: Optional[str] = None
        self.output_features: List[str] = self.model_config.architecture["output_features"]
        self.training_history: Optional[History] = None

    def _initialize_model(self) -> tf.keras.Model:
        """
        Initializes the model based on the architecture configuration.

        Returns:
            tf.keras.Model: The initialized TensorFlow model.
        """
        inputs = tf.keras.Input(
            shape=(self.model_config.architecture["input_size"],))
        x = inputs

        # Process all layers except the last one (it's the output layer)
        layers_config = self.model_config.architecture["layers"]
        for layer_config in layers_config[:-1]:
            if layer_config["type"] == "Dense":
                if layer_config.get("l2_reg", None) is not None:
                    kernel_regularizer = tf.keras.regularizers.l2(
                        layer_config.get("l2_reg", None))
                else:
                    kernel_regularizer = None
                x = tf.keras.layers.Dense(
                    units=layer_config["units"],
                    activation=layer_config.get("activation", None) if layer_config.get(
                        "activation", None) != "None" else None,
                    kernel_regularizer=kernel_regularizer
                )(x)
            elif layer_config["type"] == "BatchNormalization":
                x = tf.keras.layers.BatchNormalization(
                    axis=-1,
                    momentum=layer_config.get("momentum", 0.99),
                    epsilon=layer_config.get("epsilon", 0.001),
                    center=layer_config.get("center", True),
                    scale=layer_config.get("scale", True),
                )(x)
            elif layer_config["type"] == "Dropout":
                if "rate" not in layer_config:
                    raise ValueError(
                        "Dropout layer requires a 'rate' key in the configuration."
                    )
                x = tf.keras.layers.Dropout(rate=layer_config["rate"])(x)
            elif layer_config["type"] == "Embedding":
                if (
                    "input_dimension" not in layer_config
                    or "output_dimension" not in layer_config
                ):
                    raise ValueError(
                        "Embedding layer requires 'input_dimension' and 'output_dimension' keys in the configuration."
                    )
                x = tf.keras.layers.Embedding(
                    input_dim=layer_config["input_dimension"],
                    output_dim=layer_config["output_dimension"],
                )(x)
            else:
                raise ValueError(
                    f"Layer type '{layer_config['type']}' is not implemented."
                )

        # Process output layer
        output_config = layers_config[-1]
        if output_config["type"] == "Dense":
            outputs = tf.keras.layers.Dense(
                # Use output size from config
                units=self.model_config.architecture["output_size"],
                activation=output_config.get("activation", None) if output_config.get(
                    "activation", None) != "None" else None,
            )(x)
        else:
            raise ValueError(
                f"Output layer type '{output_config['type']}' is not implemented."
            )

        # Compile model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Process loss and metrics from config
        loss_fxn = self.model_config.training.get("loss_function", None)
        if loss_fxn is None:
            raise ValueError(
                f"Loss function type '{loss_fxn}' is not implemented."
            )
        if loss_fxn[0] == "binary_crossentropy":
            if loss_fxn[1]:
                loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            else:
                loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        elif loss_fxn[0] == "mean_squarederror":
            loss = tf.keras.losses.MeanSquaredError()
        elif loss_fxn[0] == "mean_absolute_error":
            loss = tf.keras.losses.MeanAbsoluteError()
        elif loss_fxn[0] == "categorical_crossentropy":
            loss = tf.keras.losses.CategoricalCrossentropy()
        elif loss_fxn[0] == "mse_plus_hinge_margin_loss":
            loss = mse_plus_hinge_margin_loss(alpha=loss_fxn[2])
        else:
            raise ValueError(
                f"Loss function type '{loss_fxn['loss_function']}' is not implemented."
            )

        metric = self.model_config.training.get("metrics", None)
        if metric is not None:
            if isinstance(metric, str):
                metric = [metric]

            metrics = metric
        else:
            raise ValueError(
                f"Metric type '{metric}' is not implemented."
            )

        model.compile(
            optimizer=self.model_config.training.get("optimizer", "adam"),
            loss=loss,
            metrics=metrics,
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
        feature_array = self._extract_features(examples)

        # Ensure feature array matches expected input size
        expected_input_size = self.model_config.architecture["input_size"]
        if feature_array.shape[1] != expected_input_size:
            raise ValueError(
                f"Feature array has {feature_array.shape[1]} features, but the model expects {expected_input_size}."
            )

        features_tensor = tf.convert_to_tensor(feature_array)

        return self.model(features_tensor)

    def train(
        self,
        training_examples: List[Example],
        epochs: int,
        batch_size: int,
        validation_examples: Optional[List[Example]] = None,
    ) -> None:
        """
        Trains the model using the provided examples.

        Args:
            examples (List[Example]): A list of `Example` instances containing features and labels.
            epochs (int): Number of epochs to train the model.
            batch_size (int): Batch size to use during training.
        """
        # Extract training features dynamically excluding the output feature
        training_feature_array = self._extract_features(training_examples)

        # Ensure feature array matches expected input size
        expected_input_size = self.model_config.architecture["input_size"]
        if training_feature_array.shape[1] != expected_input_size:
            raise ValueError(
                f"Training feature array has {training_feature_array.shape[1]} features, but the model expects {expected_input_size}."
            )

        # Extract training y_labels
        training_label_array = self._extract_labels(training_examples)

        # Convert training arrays to tensors
        training_features_tensor = tf.convert_to_tensor(training_feature_array)
        training_labels_tensor = tf.convert_to_tensor(training_label_array)

        # Setup tensorboard logs with model signature first
        if self._log_dir_override:
            # If user set a custom path, use it
            log_dir = self._log_dir_override
        else:
            # Otherwise, build a default
            log_dir = "logs/fit/" + f"{self.model_config.model_signature}/" + \
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # Ensure the directory exists
        os.makedirs(log_dir, exist_ok=True)

        # Create the TensorBoard callback
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir
        )

        # Call model fit with/without validation data
        if validation_examples:
            # Extract validation features dynamically excluding the output feature
            validation_feature_array = self._extract_features(
                validation_examples)

            # Ensure feature array matches expected input size
            expected_input_size = self.model_config.architecture["input_size"]
            if validation_feature_array.shape[1] != expected_input_size:
                raise ValueError(
                    f"Validation feature array has {validation_feature_array.shape[1]} features, but the model expects {expected_input_size}."
                )

            # Extract validation y_labels
            validation_label_array = self._extract_labels(validation_examples)

            # Convert validation arrays to tensors
            validation_features_tensor = tf.convert_to_tensor(
                validation_feature_array)
            validation_labels_tensor = tf.convert_to_tensor(
                validation_label_array)

            self.training_history = self.model.fit(
                training_features_tensor,
                training_labels_tensor,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(validation_features_tensor,
                                 validation_labels_tensor),
                callbacks=[tensorboard_callback],
                shuffle=True
            )
        else:
            self.training_history = self.model.fit(
                training_features_tensor,
                training_labels_tensor,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[tensorboard_callback],
                shuffle=True
            )

    def predict(
        self, examples: List[Example], return_target_labels: Optional[bool] = False
    ) -> pd.DataFrame:
        """
        Generates predictions for the provided examples.

        Args:
            examples (List[Example]): A list of `Example` instances.
            return_target_labels (Optional[bool]): Whether to include target labels in the returned DataFrame.

        Returns:
            pd.DataFrame: The predicted output.
        """
        # Forward pass to get model output
        output_tensor = self.forward(examples)
        predictions = output_tensor.numpy()  # Convert TensorFlow tensor to NumPy array

        # Create a DataFrame with predictions for each output feature
        prediction_df = pd.DataFrame(
            predictions, columns=self.output_features  # Column names from output features
        )

        if return_target_labels:
            # Include target labels in the DataFrame
            label_array = self._extract_labels(examples)
            target_df = pd.DataFrame(label_array, columns=[
                                     f"target_{name}" for name in self.output_features])
            prediction_df = pd.concat([prediction_df, target_df], axis=1)

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

    ####################
    ## HELPER METHODS ##
    ####################

    @staticmethod
    def custom_round_sigmoid_outputs(values: tf.Tensor, threshold: float) -> tf.Tensor:
        """
        Custom function to round sigmoid outputs to 0 or 1 based on a threshold.

        Args:
            values (np.ndarray or tf.Tensor): Input values between 0 and 1.
            threshold (float): Threshold for rounding. Values >= threshold are rounded to 1, others to 0.

        Returns:
            tf.Tensor: Rounded values.
        """
        return tf.cast(values >= threshold, tf.float32)

    def _extract_features(self, examples: List[Example]) -> NDArray[np.float32]:
        """
        Extracts features dynamically excluding the output features.

        Args:
            examples (List[Example]): A list of `Example` instances.

        Returns:
            np.ndarray: Extracted feature array.
        """
        return np.array(
            [
                [
                    (
                        example.features[feature_name][0]
                        if feature_name in example.features
                        else 0.0
                    )
                    for feature_name in example.features
                    if feature_name not in self.output_features
                ]
                for example in examples
            ],
            dtype=np.float32,
        )

    def _extract_labels(self, examples: List[Example]) -> NDArray[np.float32]:
        """
        Extracts labels from the examples for multiple output features.

        Args:
            examples (List[Example]): A list of `Example` instances.

        Returns:
            np.ndarray: Extracted label array, with each column corresponding to an output feature.
        """
        return np.array(
            [
                [
                    example.features[feature_name][0] if feature_name in example.features else 0.0
                    # Loop over all output features
                    for feature_name in self.output_features
                ]
                for example in examples
            ],
            dtype=np.float32,
        )

    def get_training_history(self) -> Dict[str, List[float]]:
        """
        Gets the current training history for the model.

        Returns:
            Dict[str, list]: A dictionary containing the training and validation metrics for each epoch.
                             Keys include metrics like 'loss', 'val_loss', 'accuracy', etc.
        Raises:
            ValueError: If the training history is not available.
        """
        if self.training_history is None:
            raise ValueError(
                "Training history is not available. Train the model first.")

        # Ensure the training_history attribute has a 'history' attribute
        if not hasattr(self.training_history, "history"):
            raise ValueError(
                "Invalid training history object. Missing 'history' attribute.")

        # Cast the history attribute explicitly to the expected type
        history_dict: Dict[str, List[float]] = dict(
            self.training_history.history)
        return history_dict

    def set_tensorboard_log_dir(self, log_dir: str) -> None:
        """
        Sets a custom log directory for TensorBoard logging.
        If not set, the model will generate a default directory 
        based on model signature and current date/time.
        """
        self._log_dir_override = log_dir

    def get_tensorboard_log_dir(self) -> Optional[str]:
        """
        Returns the current custom log directory override, 
        or None if none was set.
        """
        return self._log_dir_override
