from ...modules.dataset_generator.helpers.dataset_loader import DatasetLoader
from ..interfaces.data_io_interface import DataIO
from ...modules.interfaces.factory_interface import IFactory
from ...modules.data_structures.dataset_config import DatasetConfig


class DatasetLoaderCreator:
    """
    Creates a DatasetLoader object with its arguments.
    It uses the DataIOFactory to obtain the appropriate data readers for each source.
    """

    def __init__(self, config: DatasetConfig, factory: IFactory[DataIO]):
        """
        Initializes the DatasetLoaderCreator with the dataset configuration and factory.

        Args:
            config (DatasetConfig): The dataset configuration containing sources and other info.
            factory (IFactory): The factory used to create DataIO readers.
        """
        self.config = config
        self.factory = factory

    def create_loader(self) -> DatasetLoader:
        """
        Creates a DatasetLoader object with the appropriate data readers.

        Returns:
            DatasetLoader: A DatasetLoader instance initialized with the correct data readers.
        """
        data_loaders = []

        # Iterate over each Source object in the configuration
        for source in self.config.sources:
            # Use the factory to create a DataIO reader based on the file type
            data_io: DataIO = self.factory.create(source.file_type)

            data_loaders.append(data_io)

        # Create and return the DatasetLoader with the sources and their data readers
        return DatasetLoader(data_loaders, self.config.sources)
