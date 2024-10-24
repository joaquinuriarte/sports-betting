from ..interfaces.dataset_generator_interface import IDatasetGenerator
from model_manager.model_manager import ModelManager  # TODO This should be an interface
from modules.dataset_generator.interfaces.factory_interface import IFactory
from modules.dataset_generator.interfaces.join_operator_interface import IJoinOperator
from modules.dataset_generator.interfaces.feature_processor_operator_interface import (
    IFeatureProcessorOperator,
)
from modules.dataset_generator.interfaces.data_io_interface import DataIO
from modules.dataset_generator.interfaces.strategy_interface import IDatasetGeneratorStrategy
from modules.dataset_generator.dataset_generator import DatasetGenerator
from modules.data_structures.processed_dataset import ProcessedDataset
import pandas as pd

class Processor:
    def __init__(
        self,
        config_path: str,
        model_manager: ModelManager,  #TODO This should be an interface
        data_io_factory: IFactory[DataIO],
        feature_processor_factory: IFactory[IFeatureProcessorOperator],
        join_factory: IFactory[IJoinOperator],
        strategy_factory: IFactory[IDatasetGeneratorStrategy],
    ):
        # Initialize DatasetGenerator with the necessary factories
        self.dataset_generator: IDatasetGenerator = DatasetGenerator(
            config_path=config_path,
            data_io_factory=data_io_factory,
            feature_processor_factory=feature_processor_factory,
            join_factory=join_factory,
            strategy_factory=strategy_factory,
        )

        self.model_manager = model_manager # TODO This should be an interface

    def train_model(self):
        """
        Train the model using the generated dataset.
        """
        # Step 1: Generate dataset
        processed_dataset: ProcessedDataset = self.dataset_generator.generate()

        # Step 2: Train model using ModelManager
        self.model_manager.train(processed_dataset) # TODO Build this

        # TODO Should processor deal with saving weights?

    def run_inference(self, new_data: pd.DataFrame): # TODO we could wrap this argument in a class. What logic creates it?
        """
        Use the trained model to make predictions on new data.

        Args:
            new_data (pd.DataFrame): New data for which to generate predictions.

        Returns:
            Predictions from the model.
        """
        # Use the trained model to make predictions
        predictions = self.model_manager.predict(new_data) # TODO Build this
        return predictions
