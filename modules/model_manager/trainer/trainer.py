import os
from typing import Optional
from modules.data_structures.model_dataset import ModelDataset
from ..interfaces.model_interface import IModel
import logging

logging.basicConfig(level=logging.INFO)

class Trainer:
    """
    Handles the training process for models.
    """

    def __init__(self, checkpoint_dir: Optional[str] = None):
        """
        Initializes the Trainer with optional checkpointing.

        Args:
            checkpoint_dir (Optional[str]): Directory to save training checkpoints. If None, checkpoints are not saved.
        """
        self.checkpoint_dir = checkpoint_dir

    def train(self, model: IModel, model_dataset: ModelDataset):
        """
        Trains the model using the provided dataset.

        Args:
            model (IModel): The model to be trained.
            model_dataset (ModelDataset): The dataset containing features and labels.
        """
        # Extract features and labels from model_dataset
        features, labels = [], []
        for example in model_dataset.examples:
            # Collect the features and labels from each Example object
            features.append([feature for feature in example.features])
            labels.append(example.label)
        
        # TODO not under contract
        # Get training parameters from model (assuming model is initialized with ModelConfig containing these)
        epochs = model.config.training.get("epochs", 10)  # Default to 10 if not specified
        batch_size = model.config.training.get("batch_size", 32)  # Default to 32 if not specified

        # Log training information
        logging.info(f"Training the model for {model.model_config.epochs} epochs with batch size {model.model_config.batch_size}.")

        # Loop over epochs to train and save checkpoints
        for epoch in range(epochs):
            logging.info(f"Starting epoch {epoch + 1}/{epochs}.")
       
            # Run training for this epoch (assuming model.train handles batching internally)
            model.train(model_dataset.features, model_dataset.labels, epochs=1, batch_size=batch_size)

            # Save a checkpoint after each epoch if a checkpoint directory is specified
            if self.checkpoint_dir:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
                logging.info(f"Saving checkpoint to {checkpoint_path}")
                model.save(checkpoint_path)

        logging.info("Training completed.")
