import yaml
from modules.data_structures.model_config import ModelConfig


class ConfigurationLoader:
    """
    Loads and parses the configuration file for the model.
    """
    
    def __init__(self, config_path: str):
        self.config_path = config_path

    def load_config(self) -> ModelConfig:
        """
        Loads the configuration from the YAML file and returns a ModelConfig instance.
        
        Returns:
            ModelConfig: The configuration for the model.
        """
        with open(self.config_path, 'r') as file:
            config_data = yaml.safe_load(file)

        model_data = config_data['model']
        type_name = model_data['architecture']['type']
        architecture = model_data['architecture']
        training = model_data['training']
        model_path = model_data.get('save_path', None)
        
        return ModelConfig(
            type_name=type_name,
            architecture=architecture,
            training=training,
            model_path=model_path
        )