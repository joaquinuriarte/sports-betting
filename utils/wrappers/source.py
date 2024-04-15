from dataclasses import dataclass
from typing import List, Literal
from ...modules.input_output.data_io import DataIO


@dataclass
class Source:
    """
    Represents a data source specification, encapsulating the details necessary
    to locate and select specific data from a file.

    Attributes:
        path (str): The file path or URL where the data source is located.
        columns (List[str]): The specific columns to be extracted from the data source.
        primary_key (Union[str, List[str]]): The column name or list of column names
            that make up the primary key of the data.
    """

    path: str
    columns: List[str]
    primary_key: str
    join_side: Literal['left', 'right']
    file_reader: DataIO()
