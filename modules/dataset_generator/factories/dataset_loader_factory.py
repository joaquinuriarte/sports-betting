from modules.dataset_generator.factories.data_io_factory import DataIOFactory
from modules.dataset_generator.dataset_loader import DatasetLoader

class DatasetLoaderFactory:
    """
    Factory for creating a DatasetLoader based on the sources configuration.
    """

    @staticmethod
    def create_loader(sources_config: list) -> object:
        """
        Creates a DatasetLoader instance based on the provided sources configuration.
        
        Args:
            sources_config (list): Configuration for the data sources.
        
        Returns:
            DatasetLoader: An instance of the DatasetLoader with the appropriate data readers.
        """
        data_loaders = []
        for source in sources_config:
            file_type = source['file_type']
            data_io = DataIOFactory.create_reader(file_type)
            data_loaders.append(data_io)
        return DatasetLoader(data_loaders, sources_config)