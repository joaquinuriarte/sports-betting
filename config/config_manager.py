import yaml

def load_model_config(config_path: str, model_name: str) -> dict:
    """
    Loads and returns the configuration for a specific model from a YAML file.

    This function opens a YAML configuration file specified by `config_path` and
    reads its content. It then looks for a specific model's configuration by using
    the `model_name` parameter. The function parses the YAML file into a Python
    dictionary using `yaml.safe_load` and retrieves the configuration for the given
    model name. If the model name is found in the configuration, its settings are
    returned as a dictionary. If the model name is not found, an empty dictionary
    is returned, indicating that no specific configuration exists for the requested
    model.

    Parameters:
    - config_path (str): The file path to the YAML configuration file.
    - model_name (str): The name of the model for which configuration is to be loaded.

    Returns:
    - dict: A dictionary containing the configuration for the specified model. If
      the model is not found, returns an empty dictionary.
    """
    # Open the YAML configuration file in read mode.
    with open(config_path, 'r') as file:
        # Parse the YAML file into a Python dictionary.
        config = yaml.safe_load(file)
    
    # Retrieve and return the configuration for the specified model.
    model_config = config.get(model_name, {})
    return model_config