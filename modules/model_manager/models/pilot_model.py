from model_manager import ModelManager, ModelConfig, ModelDataset
from modules.data_structures.model import Model
from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
from modules.utils.yaml_reader import load_model_config


class ModelManagerV0(ModelManager):
    def __init__(self, config_path: str, model_name: str):
        """
        Initializes ModelManagerV0 with the path to the configuration file and the model name.

        Args:
            config_path (str): Path to the YAML configuration file.
            model_name (str): The name of the model to load from the configuration file.
        """
        super().__init__()
        self.config_path = config_path
        self.model_name = model_name

    def setup_model(self, config: ModelConfig): #TODO if modelconfig is passed with a path to a model then self.training_config would break?
        if config.model_path:  # TODO Where will this sit?
            # Load model from file
            self.model = torch.load(config.model_path)
        else:
            # Load model architecture from YAML using parameters from the class
            model_config = load_model_config(self.config_path, self.model_name)

            # Initialize the model with architecture from the YAML file
            self.model = Model(model_config["architecture"])

        # Load the training parameters
        self.training_config = model_config["training"]

        # Load the save destination
        self.save_path = model_config["save_path"]

        if config.inference_mode:
            self.model.eval()  # Set the model to inference mode
        else:
            self.model.train()  # Set the model to training mode

    def train_model(self, data: ModelDataset):
        # Load optimizer and loss function from training configuration
        loss_function = getattr(nn, self.training_config["loss_function"])()
        optimizer_class = getattr(optim, self.training_config["optimizer"])
        optimizer = optimizer_class(
            self.model.parameters(), lr=self.training_config["learning_rate"]
        )

        num_epochs = self.training_config["epochs"]

        # Training loop
        for epoch in range(num_epochs):
            running_loss = 0.0
            for example in data.examples:
                # Assuming each Example's features are already tensors # TODO make sure this is correct
                inputs = torch.tensor(
                    [feature for feature in example.features.values()]
                )
                labels = torch.tensor(
                    [1.0]
                )  # TODO Example label; replace with actual labels

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}, Loss: {running_loss}")
        
        # Save the model after training

    def run_inference(self, data: ModelDataset) -> ModelDataset:
        return
