from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional, Union

import pandas as pd

# Represents a unique name for a model feature.
FeatureName = str
FeatueValue = list[Union[float, int, str]]


# Describes an individual model feature.
# Feature is a mapping from a feature name to its data.
Feature = Dict[
    FeatureName, FeatueValue
]  # 'ModelData' can be defined according to your specific needs.


@dataclass
class Example:
    features: Feature


class ModelDateset(ABC):

    @abstractmethod
    def load_from_dataframe(
        self, df: pd.DataFrame, columns_to_load: Optional[list[str]] = None
    ):
        """Loads dataframe content into dataset.

        Args:
            df: dataframe that contains data that will be loaded into self.examples.
            columns_to_load: columns that we will be added to the dataset as features. If empty,
                all columns are passed in to the dataset.

        Raises:
            KeyError: if a feature specified in columns_to_load is not a column in the input df.
        """
        pass


class InMemoryModelDataset(ModelDateset):
    """Implementation for datasets that fit in memory.

    Ideal for small datasets.
    """

    examples: List[Example]

    def __init__(self, examples: List[Example]):
        self.examples = examples

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
            raise KeyError("One or more specified features do not exist in the dataframe.")

        for _, data in df.iterrows():

            example_features = defaultdict(list)
            for feature in features:
                feature_value = data[feature]
                if (
                    not isinstance(feature_value, str)
                    and not isinstance(feature_value, int)
                    and not isinstance(feature_value, float)
                ):
                    raise ValueError(
                        f"Features must be either string, int, or float. Got type {type(feature_value)} for {feature}."
                    )
            
                example_features[feature] = [feature_value]
            examples.append(Example(example_features))

        self.examples = examples
