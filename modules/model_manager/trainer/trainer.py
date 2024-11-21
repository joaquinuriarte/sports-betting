import os
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
        models: List[IModel],
        train_datasets: List[ModelDataset],
        val_datasets: Optional[List[ModelDataset]] = None
    ) -> None:
        """
        Trains multiple models using the provided datasets.

        Args:
            models (List[IModel]): The models to be trained.
            train_datasets (List[ModelDataset]): The training datasets for each model.
            val_datasets (Optional[List[ModelDataset]]): The validation datasets for each model.
        """
        if val_datasets and len(models) != len(val_datasets):
            raise ValueError("The number of validation datasets must match the number of models.")

        if len(models) != len(train_datasets):
            raise ValueError("The number of training datasets must match the number of models.")

        for i, (model, train_dataset) in enumerate(zip(models, train_datasets)):
            # Get the corresponding validation dataset if available
            val_dataset = val_datasets[i] if val_datasets else None

            # Get training parameters from model
            training_config = model.get_training_config()
            epochs = training_config.get("epochs", 10)  # Default to 10 epochs if not specified
            batch_size = training_config.get("batch_size", 32)  # Default to 32 batch size if not specified

            # Log training information
            model_signature = training_config.get("model_signature", f"model_{i}")
            logging.info(f"Training model '{model_signature}' for {epochs} epochs with batch size {batch_size}.")

            # Loop over epochs to train and save checkpoints
            for epoch in range(epochs):
                logging.info(f"Model '{model_signature}': Starting epoch {epoch + 1}/{epochs}.")

                # Train the model for this epoch
                model.train(train_dataset.examples, epochs=1, batch_size=batch_size)

                # Optionally evaluate the model on validation data
                if val_dataset:
                    val_predictions = model.predict(val_dataset.examples)
                    
                    # TODO: Additional code to calculate metrics for validation & learning curves
                    logging.info(f"Model '{model_signature}': Completed validation for epoch {epoch + 1}.")

                # Save a checkpoint after each epoch if a checkpoint directory is specified
                if self.checkpoint_dir:
                    os.makedirs(self.checkpoint_dir, exist_ok=True)
                    checkpoint_path = os.path.join(self.checkpoint_dir, f"{model_signature}_checkpoint_epoch_{epoch + 1}.pth")
                    logging.info(f"Saving checkpoint for model '{model_signature}' to {checkpoint_path}")
                    model.save(checkpoint_path)

            logging.info(f"Training for model '{model_signature}' completed.")

        logging.info("All models have completed training.")