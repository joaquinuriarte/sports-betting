from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class Source:
    """
    Represents a data source with details necessary to locate and extract data.

    Attributes:
        path (str): File path or URL where the data source is located.
        columns (List[Dict[str, Optional[str]]]): List of columns with associated metadata.
        file_type (str): The type of file (e.g., 'csv', 'xml', 'txt').
        file_reader (DataIO): Instance of DataIO for reading the file.
    """

    path: str
    columns: List[Dict[str, Optional[str]]]  # Each column is a dictionary with metadata
    file_type: str
