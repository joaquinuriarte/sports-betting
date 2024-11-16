import os
from typing import Optional
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
        train_data: ModelDataset, 
        val_data: Optional[ModelDataset] = None
    ) -> None:
        """
        Trains the model using the provided dataset.
        
        Args:
            model (IModel): The model to be trained.
            train_data (ModelDataset): The training dataset containing features and labels.
            val_data (Optional[ModelDataset]): The validation dataset used for evaluation during training.
        """
        train_features, train_labels = self._extract_features_and_labels(train_data)
        if val_data:
            val_features, val_labels = self._extract_features_and_labels(val_data)

        # Get training parameters from model
        training_config = model.get_training_config()
        epochs = training_config["epochs"]
        batch_size = training_config["batch_size"]

        # Log training information
        logging.info(f"Training the model for {epochs} epochs with batch size {batch_size}.")

        # Loop over epochs to train and save checkpoints
        for epoch in range(epochs):
            logging.info(f"Starting epoch {epoch + 1}/{epochs}.")

            # Train the model for this epoch
            model.train(train_features, train_labels, epochs=1, batch_size=batch_size)

            # Optionally evaluate the model on validation data
            if val_data:
                val_predictions = model.predict(val_features)
                
                # TODO: Additional code to calculate metrics for validation & learning curves

            # Save a checkpoint after each epoch if a checkpoint directory is specified
            if self.checkpoint_dir:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
                logging.info(f"Saving checkpoint to {checkpoint_path}")
                model.save(checkpoint_path)

        logging.info("Training completed.")

    def _extract_features_and_labels(self, model_dataset: ModelDataset):
        features, labels = [], []
        for example in model_dataset.examples:
            feature_values = [list(attr.values())[0] for attr in example.features]
            features.append(feature_values)
            labels.append(list(example.label.values())[0])
        return features, labels
