from dataclasses import dataclass
from typing import List, Literal, Optional
from .source import Source


@dataclass
class DatasetConfig:
    """
    Configuration for constructing a dataset from multiple data sources. This class specifies
    how different data sources should be combined and any additional identifiers for the dataset.

    Attributes:
        sources (List[Source]): A list of Source objects, each specifying a data source.
        join_type (Literal["inner", "left", "right", "outer"]): The type of join to use when
            combining multiple sources. Determines how rows from different sources are merged.
        name (Optional[str]): An optional name for the dataset, which can be used for identification
            or descriptive purposes.
    """

    sources: List[Source]
    name: Optional[str] = ""
