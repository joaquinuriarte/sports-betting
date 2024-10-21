from .source import Source
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class DatasetConfig:
    """
    Configuration for constructing a dataset from multiple data sources. This class specifies
    how different data sources should be combined and processed.

    Attributes:
        sources (List[Source]): A list of Source objects specifying data sources.
        joins (List[str]): A list of join types (e.g., 'inner', 'left', 'right') to be used for merging.
        feature_processor (str): The name of the feature processor to use.
        strategy (str): The name of the dataset generation strategy to apply.
        name (Optional[str]): An optional name for the dataset.
    """
    
    sources: List[Source]
    joins: List[str]
    feature_processor: str
    strategy: str
    name: Optional[str] = ""
