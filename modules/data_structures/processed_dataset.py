from dataclasses import dataclass
import pandas as pd

@dataclass
class ProcessedDataset:
    def __init__(self, features: pd.DataFrame, labels: pd.DataFrame):
        self.features = features
        self.labels = labels