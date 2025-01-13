from typing import Tuple
from modules.processor.interfaces.split_strategy_interface import ISplitStrategy
from modules.data_structures.model_dataset import ModelDataset
import random


class RandomSplitStrategy(ISplitStrategy):
    """
    Implementation of a random split strategy.
    """

    def split(
        self, dataset: ModelDataset, split_percentage: float
    ) -> Tuple[ModelDataset, ModelDataset]:
        """
        Splits the dataset randomly into two subsets.

        Args:
            dataset (ModelDataset): The dataset to split.
            split_percentage (float): The percentage of data to allocate to the first subset.

        Returns:
            Tuple[ModelDataset, ModelDataset]: The two subsets after splitting.
        """
        if not (0 < split_percentage < 100):
            raise ValueError("split_percentage must be between 0 and 100.")

        total_examples = len(dataset.examples)
        split_size = int((split_percentage / 100) * total_examples)

        # Shuffle examples to ensure randomness
        shuffled_examples = dataset.examples.copy()
        random.shuffle(shuffled_examples)

        # Split the dataset
        first_subset = shuffled_examples[:split_size]
        second_subset = shuffled_examples[split_size:]

        first_dataset = ModelDataset(examples=first_subset)
        second_dataset = ModelDataset(examples=second_subset)

        return first_dataset, second_dataset
