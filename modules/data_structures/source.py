from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Source:
    """
    Represents a data source with details necessary to locate and extract data.

    Attributes:
        path (str): File path or URL where the data source is located.
        columns (Dict[str, Dict[str, Optional[str]]]): Dictionary of columns and their metadata.
        file_type (str): The type of file (e.g., 'csv', 'xml', 'txt').
    """

    path: str
    columns: Dict[str, Dict[str, Optional[str]]]  # Each column maps to its metadata
    file_type: str
