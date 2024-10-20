from modules.dataset_generator.factories.data_io_factory import DataIOFactory
from modules.dataset_generator.interfaces.dataset_loader_interface import IDatasetLoader


class DatasetLoaderCreator:
    """
    Creates an IDatasetLoader based on the sources configuration.
    """

    def create_loader(self, sources_config: list) -> IDatasetLoader:
        """
        Creates an IDatasetLoader instance based on the provided sources configuration.
        
        Args:
            sources_config (list): Configuration for the data sources.
        
        Returns:
            IDatasetLoader: An instance of the DatasetLoader that implements IDatasetLoader.
        """
        data_loaders = []
        for source in sources_config:
            file_type = source['file_type']
            data_io = DataIOFactory.create_reader(file_type)
            data_loaders.append(data_io)

        # Return the object that implements the IDatasetLoader interface
        return IDatasetLoader(data_loaders, sources_config)

