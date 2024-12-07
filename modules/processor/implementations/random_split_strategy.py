from typing import Tuple
from modules.processor.interfaces.split_strategy_interface import ISplitStrategy
from modules.data_structures.model_dataset import ModelDataset
import random


class RandomSplitStrategy(ISplitStrategy):
    """
    Implementation of a random split strategy.
    """

    def split(
        self, dataset: ModelDataset, train_percentage: float
    ) -> Tuple[ModelDataset, ModelDataset]:
        """
        Splits the dataset randomly into training and validation datasets.

        Args:
            dataset (ModelDataset): The dataset to split.
            train_percentage (float): The percentage of data to allocate to the training set.

        Returns:
            Tuple[ModelDataset, ModelDataset]: The training and validation datasets.
        """
        if not (0 < train_percentage < 100):
            raise ValueError("train_percentage must be between 0 and 100.")

        total_examples = len(dataset.examples)
        train_size = int((train_percentage / 100) * total_examples)

        # Shuffle examples to ensure randomness
        shuffled_examples = dataset.examples.copy()
        random.shuffle(shuffled_examples)

        # Split the dataset
        train_examples = shuffled_examples[:train_size]
        val_examples = shuffled_examples[train_size:]

        train_dataset = ModelDataset(examples=train_examples)
        val_dataset = ModelDataset(examples=val_examples)

        return train_dataset, val_dataset
