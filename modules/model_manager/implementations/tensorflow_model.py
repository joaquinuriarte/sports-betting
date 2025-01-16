import tensorflow as tf
import numpy as np
import pandas as pd
from typing import List, Optional
from numpy.typing import NDArray
import datetime
import os
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
        self.prediction_threshold: float = self.model_config.architecture[
            "prediction_threshold"
        ]

    def _initialize_model(self) -> tf.keras.Model:
        """
        Initializes the model based on the architecture configuration.

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
                    activation=layer_config.get("activation", None) if layer_config.get(
                        "activation", None) != "None" else None,
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
            elif layer_config["type"] == "linear":
                x = tf.keras.layers.Dense(
                    units=layer_config["units"],
                    activation=None  # Linear activation is the default
                )(x)
            else:
                raise ValueError(
                    f"Layer type '{layer_config['type']}' is not implemented."
                )

        # TODO: Compile output layer using info on yaml instead
        outputs = tf.keras.layers.Dense(units=1)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=self.model_config.training.get("optimizer", "adam"),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[self.model_config.training.get("metrics", "accuracy")],
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
        log_dir = "logs/fit/" + f"{self.model_config.model_signature}/" + \
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # Ensure the directory exists
        os.makedirs(log_dir, exist_ok=True)

        # Create the TensorBoard callback
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1
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

            self.model.fit(
                training_features_tensor,
                training_labels_tensor,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(validation_features_tensor,
                                 validation_labels_tensor),
                callbacks=[tensorboard_callback],
            )
        else:
            self.model.fit(
                training_features_tensor,
                training_labels_tensor,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[tensorboard_callback],
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
        output_tensor = self.forward(examples)
        predictions = tf.sigmoid(output_tensor).numpy()  # Apply sigmoid

        # Use global prediction threshold
        threshold = self.prediction_threshold
        binary_predictions = self.custom_round_sigmoid_outputs(
            predictions, threshold
        ).numpy()

        prediction_df = pd.DataFrame(
            {"predictions": binary_predictions.flatten()})

        if return_target_labels:
            label_array = self._extract_labels(examples)
            prediction_df["target_label"] = label_array

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
        Extracts features dynamically excluding the output feature.

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
                    if feature_name != self.output_features
                ]
                for example in examples
            ],
            dtype=np.float32,
        )

    def _extract_labels(self, examples: List[Example]) -> NDArray[np.float32]:
        """
        Extracts labels from the examples.

        Args:
            examples (List[Example]): A list of `Example` instances.

        Returns:
            np.ndarray: Extracted label array.
        """
        return np.array(
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

    def set_prediction_threshold(self, threshold: float) -> None:
        """
        Sets the prediction threshold for the model.

        Args:
            threshold (float): The new prediction threshold value.
        """
        self.prediction_threshold = threshold
