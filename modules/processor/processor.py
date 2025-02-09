from modules.data_structures.processed_dataset import ProcessedDataset
from modules.data_structures.model_dataset import ModelDataset, Example
from modules.interfaces.factory_interface import IFactory
from modules.processor.interfaces.split_strategy_interface import ISplitStrategy
from modules.processor.helpers.configuration_loader import ConfigurationLoader
from modules.interfaces.processor_interface import IProcessor
from typing import Tuple, Optional


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
        self.chronological_column = split_config.get(
            "chronological_column", None)
        self.train_split = split_config["train_split"]
        self.val_split = split_config["val_split"]
        self.test_split = split_config["test_split"]
        self.use_val = split_config["use_val"]
        self.use_test = split_config["use_test"]

        # Create split strategy implementation
        self.split_strategy: ISplitStrategy = split_strategy_factory.create(
            self.split_strategy_name, split_config=split_config
        )

    def generate(
        self,
    ) -> Tuple[ModelDataset, Optional[ModelDataset], Optional[ModelDataset]]:
        """
        Generates training, validation, and test datasets.
        """

        # Validate split percentages
        total_split = self.train_split + \
            (self.val_split or 0) + (self.test_split or 0)

        if total_split != 100:
            raise ValueError("Split percentages must add up to 100.")

        # Split the dataset
        # TODO Provide a way to ingest seed so split is similar if ran more than once
        train_dataset, remaining_dataset = self.split_strategy.split(
            self.processed_dataset, self.train_split
        )

        val_dataset, test_dataset = None, None
        if self.use_val and remaining_dataset is not None:
            val_split_ratio = 100 * (self.val_split /
                                     (self.val_split + self.test_split))
            val_dataset, test_dataset = self.split_strategy.split(
                remaining_dataset, val_split_ratio
            )
        elif self.use_test and remaining_dataset is not None:
            test_dataset = remaining_dataset

        # Convert each ProcessedDataset subset to a ModelDataset.
        train_dataset = self.build_model_dataset(train_dataset)
        val_dataset = self.build_model_dataset(
            val_dataset) if val_dataset is not None else None
        test_dataset = self.build_model_dataset(
            test_dataset) if test_dataset is not None else None

        return train_dataset, val_dataset, test_dataset

    def build_model_dataset(self, processed_dataset: ProcessedDataset) -> ModelDataset:
        """
        Converts a ProcessedDataset into a ModelDataset by extracting examples.
        Prior to conversion, the chronological column (if present) is dropped,
        as it is not intended to be used as a feature.
        """
        examples = []
        # Create a copy of the DataFrame to avoid modifying the original.
        df = processed_dataset.features.copy()

        # If the chronological column is set and exists in the DataFrame, drop it.
        if self.chronological_column is not None and self.chronological_column in df.columns:
            df = df.drop(columns=[self.chronological_column])

        # Iterate through the rows of the cleaned DataFrame.
        for game_id, feature_row in df.iterrows():
            assert game_id is not None, "Game ID cannot be None"

            # Create the feature dictionary, wrapping each value in a list.
            example_features = {
                str(feature_name): [feature_value]
                for feature_name, feature_value in feature_row.items()
            }

            # Create an Example object and add it to the list.
            examples.append(Example(features=example_features))

        return ModelDataset(examples=examples)
