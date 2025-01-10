from modules.data_structures.processed_dataset import ProcessedDataset
from modules.interfaces.factory_interface import IFactory
from modules.processor.interfaces.split_strategy_interface import ISplitStrategy
from modules.processor.helpers.configuration_loader import ConfigurationLoader
from modules.interfaces.processor_interface import IProcessor
from modules.data_structures.model_dataset import ModelDataset, Example
from typing import Tuple, Optional, List, Any


class Processor(IProcessor):
    """
    Main orchestrator for ModelDataset generation and train/test split.
    """

    def __init__(
        self,
        yaml_path: str,
        configuration_loader: ConfigurationLoader,
        processed_dataset: ProcessedDataset,
        split_strategy_factory: IFactory[ISplitStrategy],
    ) -> None:

        self.processed_dataset = processed_dataset

        # Load split strategy and parameters
        split_config = configuration_loader.load_config(yaml_path)
        self.split_strategy_name = split_config["strategy"]
        self.train_split = split_config["train_split"]
        self.val_split = split_config["val_split"]
        self.test_split = split_config["test_split"]
        self.use_val = split_config["use_val"]
        self.use_test = split_config["use_test"]

        # Create split strategy implementation
        self.split_strategy: ISplitStrategy = split_strategy_factory.create(
            self.split_strategy_name)

    def generate(
        self
    ) -> Tuple[ModelDataset, Optional[ModelDataset], Optional[ModelDataset]]:
        """
        Generates training, validation, and test datasets.
        """
        # Create Model Dataset
        model_dataset = self.build_model_dataset(self.processed_dataset)

        # Validate split percentages
        total_split = self.train_split + \
            (self.val_split or 0) + (self.test_split or 0)
        if total_split != 100:
            raise ValueError("Split percentages must add up to 100.")

        # Split the dataset
        train_dataset, remaining_dataset = self.split_strategy.split(
            model_dataset, self.train_split / 100.0
        )

        val_dataset, test_dataset = None, None
        if self.use_val and remaining_dataset is not None:
            val_split_ratio = self.val_split / \
                (self.val_split + self.test_split)
            val_dataset, test_dataset = self.split_strategy.split(
                remaining_dataset, val_split_ratio
            )
        elif self.use_test and remaining_dataset is not None:
            test_dataset = remaining_dataset

        return train_dataset, val_dataset, test_dataset

    def build_model_dataset(self, processed_dataset: ProcessedDataset) -> ModelDataset:
        """
        Converts a ProcessedDataset into a ModelDataset by extracting examples.

        Args:
            processed_dataset (ProcessedDataset): The processed dataset containing features.

        Returns:
            ModelDataset: A dataset with examples containing features and labels combined.
        """
        examples = []

        # Iterate through the rows of features in processed_dataset
        for game_id, feature_row in processed_dataset.features.iterrows():
            assert game_id is not None, "Game ID cannot be None"

            # Create the feature dictionary
            example_features = {
                str(feature_name): [feature_value]
                for feature_name, feature_value in feature_row.items()
            }

            # Create an Example object and add it to the list
            examples.append(Example(features=example_features))

        return ModelDataset(examples=examples)
