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
        Loads the model configuration from the YAML file.
        
        Returns:
            ModelConfig: A dataclass representing the configuration for the model.
        """
        with open(self.config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        
        model_architecture = config_data["model"]["architecture"]
        training_config = config_data["model"]["training"]
        save_path = config_data["model"]["save_path"]

        return ModelConfig(
            model_path=save_path,
            architecture_config=model_architecture,
            training_epochs=training_config.get("epochs"),
            learning_rate=training_config.get("learning_rate"),
            optimizer=training_config.get("optimizer"),
            loss_function=training_config.get("loss_function")
        )
