from modules.dataset_generator.operations.feature_processing_operations import (
    TopNPlayersFeatureProcessor,
)
from modules.dataset_generator.interfaces.feature_processor_operator_interface import (
    IFeatureProcessorOperator,
)
from modules.dataset_generator.interfaces.factory_interface import IFactory


class FeatureProcessorFactory(IFactory):
    """
    Factory for creating feature processors based on the type specified.
    """

    @staticmethod
    def create_processor(processing_type: str) -> IFeatureProcessorOperator:
        """
        Creates a FeatureProcessor instance based on the provided processing type.

        Args:
            processing_type (str): Type of the feature processing (e.g., 'top_n_players').

        Returns:
            FeatureProcessor: An instance of the appropriate FeatureProcessor.
        """
        if processing_type == "top_n_players":
            feature_processor: IFeatureProcessorOperator = TopNPlayersFeatureProcessor()
            return feature_processor
        else:
            raise ValueError(f"Unsupported feature processing type: {processing_type}")
