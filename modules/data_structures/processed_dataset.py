from dataclasses import dataclass
import pandas as pd


@dataclass
class ProcessedDataset:
    """
    Represents a processed dataset with features and labels.

    Attributes:
        features (pd.DataFrame): The feature data.
    """

    features: pd.DataFrame
