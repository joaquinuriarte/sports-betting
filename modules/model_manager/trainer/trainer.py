import os
import pandas as pd
import numpy as np
from modules.data_structures.model_dataset import Example
from typing import Optional, List
from modules.data_structures.model_dataset import ModelDataset
from ..interfaces.model_interface import IModel
from ..interfaces.trainer_interface import ITrainer

import logging

logging.basicConfig(level=logging.INFO)


class Trainer(ITrainer):
    """
    Handles the training process for models.
    """

    def __init__(self, checkpoint_dir: Optional[str] = None) -> None:
        """
        Initializes the Trainer with optional checkpointing.

        Args:
            checkpoint_dir (Optional[str]): Directory to save training checkpoints. If None, checkpoints are not saved.
        """
        self.checkpoint_dir = checkpoint_dir

    def train(
        self,
        model: IModel,
        train_dataset: ModelDataset,
        val_dataset: Optional[ModelDataset] = None,
    ) -> None:
        """
        Trains a model using the provided datasets.

        Args:
            model (IModel): The model to be trained.
            train_dataset (ModelDataset): The training dataset containing features and labels.
            val_dataset (Optional[ModelDataset]): The validation dataset used for evaluation during training.
        """
        # Get training parameters from model
        model_config = model.get_training_config()
        epochs = model_config.training.get("epochs", 10)
        batch_size = model_config.training.get("batch_size", 32)
        output_features = model_config.architecture["output_features"]

        # Log training information
        model_signature = model_config.model_signature
        logging.info(
            f"Training model '{model_signature}' for {epochs} epochs with batch size {batch_size}."
        )

        # Loop over epochs to train and save checkpoints
        for epoch in range(epochs):
            logging.info(
                f"Model '{model_signature}': Starting epoch {epoch + 1}/{epochs}."
            )

            # Train the model for this epoch
            model.train(train_dataset.examples,
                        epochs=1, batch_size=batch_size)

            logging.info(
                f"Model '{model_signature}': Completed epoch {epoch + 1}.")

        # Evaluate on validation data after training
        if val_dataset:
            val_predictions = model.predict(val_dataset.examples)
            true_labels = self.extract_labels(
                val_dataset.examples, output_features)

            # Calculate validation accuracy
            accuracy = self.calculate_accuracy(val_predictions, true_labels)
            logging.info(
                f"Model '{model_signature}': Validation Accuracy after training: {accuracy:.4f}"
            )

        # Save a checkpoint after each epoch if a checkpoint directory is specified
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"{model_signature}_checkpoint_epoch_{epoch + 1}.pth",
            )
            logging.info(
                f"Saving checkpoint for model '{model_signature}' to {checkpoint_path}"
            )
            model.save(checkpoint_path)

        logging.info(f"Training for model '{model_signature}' completed.")

    def extract_labels(self, examples: List[Example], output_features: str) -> np.ndarray:
        """
        Extracts true labels from examples based on the specified output feature.

        Args:
            examples (List[Example]): A list of `Example` instances.
            output_features (str): The name of the feature containing the labels.

        Returns:
            np.ndarray: Array of true labels.
        """
        label_array = np.array(
            [
                (
                    example.features[output_features][0]
                    if output_features in example.features
                    else 0.0
                )
                for example in examples
            ],
            dtype=np.float32,
        )
        return label_array

    def calculate_accuracy(self, predictions: pd.DataFrame, true_labels: np.ndarray) -> float:
        """
        Calculates the accuracy of predictions compared to true labels.

        Args:
            predictions (pd.DataFrame): Predictions from the model.
            true_labels (np.ndarray): Ground truth labels.

        Returns:
            float: Accuracy value.
        """
        predicted_classes = predictions.values.argmax(axis=1)
        true_classes = true_labels.astype(int)
        accuracy = (predicted_classes == true_classes).mean()
        return accuracy
