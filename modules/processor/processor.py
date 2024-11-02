from ..interfaces.dataset_generator_interface import IDatasetGenerator
from ..interfaces.model_manager_interface import IModelManager
from modules.interfaces.factory_interface import IFactory
from modules.dataset_generator.interfaces.join_operator_interface import IJoinOperator
from modules.dataset_generator.interfaces.feature_processor_operator_interface import (
    IFeatureProcessorOperator,
)
from modules.dataset_generator.interfaces.data_io_interface import DataIO
from modules.dataset_generator.interfaces.strategy_interface import (
    IDatasetGeneratorStrategy,
)
from modules.dataset_generator.dataset_generator import DatasetGenerator
from modules.data_structures.processed_dataset import ProcessedDataset
import pandas as pd

# TODO Processor tiene que crear ModelDataset from ProcessedDataset. Using info on yaml, create tensors or any other datatype required for the model. This way we decouple this from model manager and we can train many model configurations on the same dataset without having to redundantly repeat the processedDataset->Modeldataset conversion


class Processor:
    def __init__(
        self,
        config_path: str,
        model_manager: IModelManager,
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

        self.config_path = config_path
        self.model_manager: IModelManager = (
            model_manager  # TODO we need to construct instance here. What parameters does it accept? Overlap with step 2 below
        )

    def train_model(self):
        """
        Train the model using the generated dataset.
        """
        # Step 1: Generate dataset
        processed_dataset: ProcessedDataset = self.dataset_generator.generate()

        # Step 2: Set up the model
        self.model_manager.setup_model(
            self.config_path
        )  # TODO we need to build this. This should create datastructures, tensors, ect?

        # Step 3: Train model using ModelManager
        self.model_manager.train(processed_dataset.features, processed_dataset.labels)

        # TODO Should processor deal with saving weights?
        # Step 4: Save model? Or is that done directly by train?

    def run_inference(
        self, new_data: pd.DataFrame
    ):  # TODO we could wrap this argument in a class. What logic creates it?
        """
        Use the trained model to make predictions on new data.

        Args:
            new_data (pd.DataFrame): New data for which to generate predictions.

        Returns:
            Predictions from the model.
        """
        # Use the trained model to make predictions
        predictions = self.model_manager.predict(new_data)  # TODO Build this
        return predictions
