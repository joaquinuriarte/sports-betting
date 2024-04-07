from dataclasses import dataclass
from typing import List

@dataclass
class Source:
    """
    Represents a data source specification, encapsulating the details necessary
    to locate and select specific data from a file.

    Attributes:
        path (str): The file path or URL where the data source is located.
        columns (List[str]): The specific columns to be extracted from the data source.
    """
    path: str
    columns: List[str]
