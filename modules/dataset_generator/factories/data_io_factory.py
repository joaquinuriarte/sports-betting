from modules.dataset_generator.helpers.data_readers import CsvIO, TxtIO, XmlIO
from modules.dataset_generator.interfaces.data_io_interface import DataIO
from modules.interfaces.factory_interface import IFactory
from typing import Any


class DataIOFactory(IFactory[DataIO]):
    """
    Factory for creating data readers based on file type.
    """

    @staticmethod
    def create(type_name: str, *args: Any, **kwargs: Any) -> DataIO:
        """
        Creates a DataIO reader instance based on the provided file type.

        Args:
            file_type (str): Type of the file (e.g., 'csv', 'xml', 'txt').

        Returns:
            DataIO: An instance of the appropriate DataIO reader.
        """
        if type_name == "csv":
            return CsvIO()
        elif type_name == "xml":
            return XmlIO()
        elif type_name == "txt":
            return TxtIO()
        else:
            raise ValueError(f"Unsupported file type: {type_name}")
