from modules.dataset_generator.operations.data_readers import CsvIO, TxtIO, XmlIO

class DataIOFactory:
    """
    Factory for creating data readers based on file type.
    """

    @staticmethod
    def create_reader(file_type: str) -> object:
        """
        Creates a DataIO reader instance based on the provided file type.
        
        Args:
            file_type (str): Type of the file (e.g., 'csv', 'xml', 'txt').
        
        Returns:
            DataIO: An instance of the appropriate DataIO reader.
        """
        if file_type == 'csv':
            return CsvIO()
        elif file_type == 'xml':
            return XmlIO()
        elif file_type == 'txt':
            return TxtIO()
        else:
            raise ValueError(f"Unsupported file type: {file_type}")