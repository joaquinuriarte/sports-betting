from .interfaces.dataset_loader_interface import IDatasetLoader
from .interfaces.strategy_interface import IDatasetGeneratorStrategy
from modules.data_structures.processed_dataset import ProcessedDataset
from ..interfaces.dataset_generator_interface import IDatasetGenerator


class DatasetGenerator(IDatasetGenerator):
    """
    Main orchestrator for dataset generation.
    """

    def __init__(
        self,
        dataset_loader: IDatasetLoader,
        dataset_strategy: IDatasetGeneratorStrategy,
    ):
        self.dataset_loader = dataset_loader
        self.dataset_strategy = dataset_strategy

    def generate(self) -> ProcessedDataset:
        """
        Generate the dataset by loading data, optionally joining, and processing features.

        Returns:
            ProcessedDataset: The generated dataset containing features and labels.
        """
        # Load the data sources
        dataframes = self.dataset_loader.load_data()

        # Generate the features and labels using the strategy
        processed_dataset: ProcessedDataset = self.dataset_strategy.generate(dataframes)

        return processed_dataset
