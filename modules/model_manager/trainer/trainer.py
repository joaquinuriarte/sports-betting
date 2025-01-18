from typing import Optional, Tuple, List
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

        # Log training information
        model_signature = model_config.model_signature
        logging.info(
            f"Training model '{model_signature}' for {epochs} epochs with batch size {batch_size}."
        )

        # Train the model for this epoch
        if val_dataset:
            model.train(
                train_dataset.examples,
                epochs=epochs,
                batch_size=batch_size,
                validation_examples=val_dataset.examples,
            )
        else:
            model.train(train_dataset.examples,
                        epochs=epochs, batch_size=batch_size)

        # Final print
        logging.info(f"Model '{model_signature}': Finished training.")
