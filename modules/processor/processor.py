from modules.data_structures.processed_dataset import ProcessedDataset
from modules.interfaces.factory_interface import IFactory
from modules.processor.interfaces.split_strategy_interface import ISplitStrategy #TODO Create


class Processor(): #TODO Create and add interface for processor
    """
    Main orchestrator for ModelDataset generation and train/test split.
    """

    def __init__(
        self,
        processed_dataset: ProcessedDataset,
        split_strategy_factory: IFactory[ISplitStrategy], #TODO Create and add interface for split strategy
    ) -> None:

        self.processed_dataset = processed_dataset

        # Create split strategy implementation
        self.split_strategy: ISplitStrategy

## Responsabilities
    # Convert processed dataset (output of dataset generator) into Model Dataset (new DS that ricky created)
    # Use a factory to create instance of split class and split model dataset into training ds and validation ds
