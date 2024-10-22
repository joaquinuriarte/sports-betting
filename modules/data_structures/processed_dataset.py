from dataclasses import dataclass
import pandas as pd


@dataclass
class ProcessedDataset:
    """
    Represents a processed dataset with features and labels.

    Attributes:
        features (pd.DataFrame): The feature data.
        labels (pd.DataFrame): The label data.
    """

    features: pd.DataFrame
    labels: pd.DataFrame
