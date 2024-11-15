from dataclasses import dataclass
from typing import List
from main.interfaces.data_io_interface import DataIO


@dataclass
class Source:
    """
    Represents a data source with details necessary to locate and extract data.

    Attributes:
        path (str): File path or URL where the data source is located.
        columns (List[str]): List of columns to extract.
        file_type (str): The type of file (e.g., 'csv', 'xml', 'txt').
        file_reader (DataIO): Instance of DataIO for reading the file.
    """

    path: str
    columns: List[str]
    file_type: str
