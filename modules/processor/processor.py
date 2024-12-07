import pandas as pd
from collections import defaultdict
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
        split_strategy_name, self.percent_split = configuration_loader.load_config(yaml_path)

        # Create split strategy implementation
        self.split_strategy: ISplitStrategy = split_strategy_factory.create(split_strategy_name)
    
    def generate(self, val_dataset_flag: Optional[bool] = True) -> Tuple[ModelDataset, Optional[ModelDataset]]:

        # Create Model Dataset
        train_dataset = self.build_model_dataset(self.processed_dataset)
        validation_dataset = None

        # Split Model Dataset if flag is True
        if val_dataset_flag:
            train_dataset, validation_dataset = self.split_strategy.split(train_dataset, self.percent_split)

        return train_dataset, validation_dataset

    def build_model_dataset(
        self, processed_dataset: ProcessedDataset
    ) -> ModelDataset:
        """
        Converts a ProcessedDataset into a ModelDataset by merging features and labels.

        Args:
            processed_dataset (ProcessedDataset): The processed dataset containing features and labels.

        Returns:
            ModelDataset: A dataset with examples containing both features and labels.
        """
        examples = []

        # Iterate through the rows of features and labels
        for game_id, feature_row in processed_dataset.features.iterrows():
            label_row = processed_dataset.labels.loc[game_id]

            # Create the feature dictionary
            example_features = {
                feature_name: [feature_value]
                for feature_name, feature_value in feature_row.items()
            }

            # Append labels to the feature dictionary
            example_features["PTS_home"] = [label_row["PTS_home"]]
            example_features["PTS_away"] = [label_row["PTS_away"]]

            # Create an Example object and add it to the list
            examples.append(Example(features=example_features))

        # Return a ModelDataset containing all examples
        return ModelDataset(examples=examples)

    
    def load_from_dataframe(
        self, df: pd.DataFrame, columns_to_load: Optional[List[str]] = None 
    ):
        """Loads dataframe content into dataset.

        Args:
            df: dataframe that contains data that will be loaded into self.examples.
            columns_to_load: columns that we will be added to the dataset as features. If empty,
                all columns are passed in to the dataset.

        Raises:
            KeyError: if a feature specified in columns_to_load is not a column in the input df.
        """
        examples = []
        features = columns_to_load if columns_to_load else df.columns

        if set(features).intersection(set(df.columns)) < set(features):
            raise KeyError(
                "One or more specified features do not exist in the dataframe."
            )

        for _, data in df.iterrows():
            example_features = defaultdict(list)
            for feature in features:
                feature_value = data[feature]
                print("Feature: ", feature_value)
                if (
                    isinstance(feature_value, str)
                    or isinstance(feature_value, int)
                    or isinstance(feature_value, float)
                ):
                    example_features[feature] = [feature_value]
                elif isinstance(feature_value, list) and (
                    self.check_list_is_type(feature_value, float)
                    or self.check_list_is_type(feature_value, int)
                    or self.check_list_is_type(feature_value, str)
                ):
                    example_features[feature] = feature_value
                
                else:
                    raise TypeError(
                        "Features must be string, ints, floats, or a list of"
                        "strings, ints, or floats."
                    )
            examples.append(Example(example_features))

        self.examples = examples

    def check_list_is_type(input: List[Any], instance: Any) -> bool:
        return all(isinstance(x, instance) for x in input)