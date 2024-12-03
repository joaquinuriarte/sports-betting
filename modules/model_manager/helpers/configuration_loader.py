import yaml
import hashlib
from modules.data_structures.model_config import ModelConfig

class ConfigurationLoader:
    """
    Loads and parses the configuration file for the model.
    """

    def load_config(self, config_path: str) -> ModelConfig:
        """
        Loads the configuration from the YAML file and returns a ModelConfig instance.

        Args:
            config_path (str): Path to the YAML configuration file.

        Returns:
            ModelConfig: The configuration for the model.
        """
        with open(config_path, "r") as file:
            config_data = yaml.safe_load(file)

        # Check if model signature exists
        signature = config_data.get("model", {}).get("model_signature")
        if not signature:
            # Generate a signature by hashing the entire YAML configuration
            config_str = yaml.dump(config_data)
            signature = hashlib.md5(config_str.encode()).hexdigest()

            # Add the generated signature to the configuration
            config_data["model"]["model_signature"] = signature
            self.update_config(config_path, "model.model_signature", signature)


        # Parse the updated configuration into a ModelConfig object
        model_data = config_data["model"]
        type_name = model_data["architecture"]["type"]
        architecture = model_data["architecture"]
        training = model_data["training"]

        return ModelConfig(
            type_name=type_name,
            architecture=architecture,
            training=training,
            model_signature=signature,
        )

    def update_config(
        self, yaml_file_path: str, field_name: str, new_value: str
    ) -> None:
        """
        Updates a specific field in the YAML configuration.

        Args:
            yaml_file_path (str): Path to the YAML file to update.
            field_name (str): Dot-separated field name to update.
            new_value: The new value to assign to the field.
        """
        with open(yaml_file_path, "r") as config_file:
            config_data = yaml.safe_load(config_file)

        # Update the desired field using dot-separated keys
        keys = field_name.split(".")
        config_part = config_data
        for key in keys[:-1]:
            config_part = config_part.setdefault(key, {})

        # Update the final field
        config_part[keys[-1]] = new_value

        # Save the updated configuration back to the YAML file
        with open(yaml_file_path, "w") as config_file:
            yaml.dump(config_data, config_file)
