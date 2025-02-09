from typing import Tuple
from modules.processor.interfaces.split_strategy_interface import ISplitStrategy
from modules.data_structures.processed_dataset import ProcessedDataset
from modules.data_structures.processed_dataset import ProcessedDataset


class RandomSplitStrategy(ISplitStrategy):
    """
    Splits a ProcessedDataset randomly into two subsets.
    """

    def split(
        self, dataset: ProcessedDataset, split_percentage: float
    ) -> Tuple[ProcessedDataset, ProcessedDataset]:
        """
        Randomly shuffles and splits the dataset.
        """
        total_examples = len(dataset.features)
        split_size = int((split_percentage / 100) * total_examples)

        # Shuffle the DataFrame rows randomly.
        df_shuffled = dataset.features.sample(frac=1).reset_index(drop=True)

        train_df = df_shuffled.iloc[:split_size].copy()
        remaining_df = df_shuffled.iloc[split_size:].copy()

        return ProcessedDataset(features=train_df), ProcessedDataset(features=remaining_df)
