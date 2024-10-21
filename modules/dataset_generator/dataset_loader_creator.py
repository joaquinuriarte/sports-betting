from modules.dataset_generator.factories.data_io_factory import DataIOFactory
from modules.dataset_generator.dataset_loader import DatasetLoader
from modules.dataset_generator.interfaces.data_io_interface import DataIO
from modules.dataset_generator.interfaces.factory_interface import IFactory


class DatasetLoaderCreator:
    """
    Creates a DatasetLoader object with its arguments.
    It also calls the DataIOFactory to obtain the appropriate data readers for each source.
    """
    def __init__(self, sources_config: list, factory: IFactory):
        self.factory = factory
        self.sources_config = sources_config

    def create_loader(self) -> DatasetLoader:
        """
        Creates a DatasetLoader object with its arguments based on the provided sources configuration.
        This involves calling the DataIOFactory to get the appropriate readers for each data source.
        
        Args:
            sources_config (list): Configuration for the data sources.
        
        Returns:
            DatasetLoader: A DatasetLoader instance with the appropriate data readers.
        """
        data_loaders = []
        for source in self.sources_config:
            file_type = source['file_type']
            data_io: DataIO = self.factory.create(file_type)
            data_loaders.append(data_io)

        # Return the object that implements the IDatasetLoader interface
        return DatasetLoader(data_loaders, self.sources_config)

