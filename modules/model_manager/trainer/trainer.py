import os
import pandas as pd
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import Optional, List
from modules.data_structures.model_dataset import ModelDataset, Example
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
        self.train_accuracies = []
        self.val_accuracies = []

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

        # Prepare real-time plotting
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_title("Training and Validation Accuracy")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")
        train_line, = ax.plot([], [], label="Training Accuracy", color="blue")
        val_line, = ax.plot(
            [], [], label="Validation Accuracy", color="orange")
        ax.legend()

        # Loop over epochs to train and save checkpoints
        for epoch in range(epochs):
            logging.info(
                f"Model '{model_signature}': Starting epoch {epoch + 1}/{epochs}."
            )

            # Train the model for this epoch
            model.train(train_dataset.examples,
                        epochs=1, batch_size=batch_size)

            # Calculate training accuracy
            train_predictions = model.predict(train_dataset.examples)
            train_labels = self.extract_labels(
                train_dataset.examples, output_features)
            train_accuracy = self.calculate_accuracy(
                train_predictions, train_labels)
            self.train_accuracies.append(train_accuracy)

            # Calculate validation accuracy
            if val_dataset:
                val_predictions = model.predict(val_dataset.examples)
                val_labels = self.extract_labels(
                    val_dataset.examples, output_features)
                val_accuracy = self.calculate_accuracy(
                    val_predictions, val_labels)
                self.val_accuracies.append(val_accuracy)
                logging.info(
                    f"Epoch {epoch + 1}/{epochs} - Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}"
                )
            else:
                logging.info(
                    f"Epoch {epoch + 1}/{epochs} - Training Accuracy: {train_accuracy:.4f}")

            # Update the plot
            train_line.set_xdata(range(1, len(self.train_accuracies) + 1))
            train_line.set_ydata(self.train_accuracies)
            if val_dataset:
                val_line.set_xdata(range(1, len(self.val_accuracies) + 1))
                val_line.set_ydata(self.val_accuracies)
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.01)

        # Save the final plot
        plt.ioff()
        plt.savefig(
            f"{self.checkpoint_dir}/{model_signature}_training_validation_accuracy.png")
        plt.show()

    def extract_labels(self, examples: List[Example], output_features: str) -> NDArray[np.float32]:
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

    def calculate_accuracy(self, predictions: pd.DataFrame, true_labels: NDArray[np.float32]) -> float:
        predicted_classes = predictions.values.argmax(axis=1)
        true_classes = true_labels.astype(int)
        accuracy: float = (predicted_classes == true_classes).mean()
        return accuracy
