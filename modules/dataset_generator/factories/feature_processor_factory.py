from modules.dataset_generator.operations.feature_processing_operations import TopNPlayersFeatureProcessor

class FeatureProcessorFactory():
    """
    Factory for creating feature processors based on the type specified.
    """

    @staticmethod
    def create_processor(processing_type: str) -> object: #TODO: Should return interface object
        """
        Creates a FeatureProcessor instance based on the provided processing type.
        
        Args:
            processing_type (str): Type of the feature processing (e.g., 'top_n_players').
        
        Returns:
            FeatureProcessor: An instance of the appropriate FeatureProcessor.
        """
        if processing_type == 'top_n_players':
            return TopNPlayersFeatureProcessor()
        else:
            raise ValueError(f"Unsupported feature processing type: {processing_type}")