from .source import Source
from dataclasses import dataclass
from typing import List, TypedDict, Dict, Any
from modules.dataset_generator.interfaces.join_operator_interface import IJoinOperator

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
    joins: List[Dict[str, Any]]
    strategy: str
    feature_processor_type: str
    top_n_players: int
    sorting_criteria: str
    player_stats_columns: List[str]


class JoinOperation(TypedDict):
    """
    TypedDict representing a join operation and its associated keys.
    """

    operator: IJoinOperator
    keys: List[str]
