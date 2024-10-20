from modules.dataset_generator.factories.data_io_factory import DataIOFactory
from modules.dataset_generator.dataset_loader import DatasetLoader

class DatasetLoaderCreator:
    """
    Creates a DatasetLoader based on the sources configuration.
    """

    def create_loader(self, sources_config: list) -> DatasetLoader:
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