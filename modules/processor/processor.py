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

        # Find split strategy
        split_strategy_name, self.percent_split = configuration_loader.load_config(
            yaml_path
        )

        # Create split strategy implementation
        self.split_strategy: ISplitStrategy = split_strategy_factory.create(
            split_strategy_name
        )

    def generate(
        self, val_dataset_flag: Optional[bool] = True
    ) -> Tuple[ModelDataset, Optional[ModelDataset]]:
        """
        Generates training and optionally validation datasets.
        """
        # Create Model Dataset
        train_dataset = self.build_model_dataset(self.processed_dataset)
        validation_dataset = None

        # Split Model Dataset if flag is True
        if val_dataset_flag:
            if self.percent_split is None:
                raise ValueError("percent_split must be defined in the configuration.")
            train_dataset, validation_dataset = self.split_strategy.split(
                train_dataset, float(self.percent_split)
            )

        return train_dataset, validation_dataset

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
