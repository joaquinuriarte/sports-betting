from .source import Source
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class DatasetConfig:
    """
    Configuration for constructing a dataset from multiple data sources. This class specifies
    how different data sources should be combined and processed.

    Attributes:
        sources (List[Source]): A list of Source objects in the order they should be merged.
        joins (List[str]): A list of join types (e.g., 'inner', 'left') for each merge operation.
        feature_processor (str): The name of the feature processor to use.
        strategy (str): The name of the strategy for dataset generation.
        name (Optional[str]): Optional name for the dataset.
    """

    sources: List[Source]
    joins: List[str]
    feature_processor: str
    strategy: str
    name: Optional[str] = ""
