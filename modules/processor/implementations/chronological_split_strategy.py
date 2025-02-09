from typing import Tuple
from modules.processor.interfaces.split_strategy_interface import ISplitStrategy
from modules.data_structures.processed_dataset import ProcessedDataset
from modules.data_structures.processed_dataset import ProcessedDataset


class ChronologicalSplitStrategy(ISplitStrategy):
    """
    Splits a ProcessedDataset chronologically using a specified date column.
    """

    def __init__(self, chronological_column: str):
        self.chronological_column = chronological_column

    def split(
        self, dataset: ProcessedDataset, split_percentage: float
    ) -> Tuple[ProcessedDataset, ProcessedDataset]:
        """
        Splits the dataset into two subsets based on the chronological order.
        The training subset contains the oldest records up to the specified percentage.
        """
        # Sort the DataFrame by the chronological column in ascending order (oldest first)
        sorted_df = dataset.features.sort_values(
            by=self.chronological_column, ascending=True
        ).reset_index(drop=True)

        total_examples = len(sorted_df)
        split_size = int((split_percentage / 100) * total_examples)

        train_df = sorted_df.iloc[:split_size].copy()
        remaining_df = sorted_df.iloc[split_size:].copy()

        train_df.drop(columns=[self.chronological_column], inplace=True)
        remaining_df.drop(columns=[self.chronological_column], inplace=True)

        return ProcessedDataset(features=train_df), ProcessedDataset(features=remaining_df)
