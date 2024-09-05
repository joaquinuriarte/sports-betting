from model_manager import ModelManager, ModelConfig, ModelDataset
from utils.wrappers.model import Model
from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
from configuration.config_manager import load_model_config

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

    def setup_model(self, config: ModelConfig):
        if config.model_path: #TODO Where will this sit?
            # Load model from file
            self.model = torch.load(config.model_path)
        else:
            # Load model architecture from YAML using parameters from the class
            model_config = load_model_config(self.config_path, self.model_name)
            
            # Initialize the model with architecture from the YAML file
            self.model = Model(model_config['architecture'])

        if config.inference_mode:
            self.model.eval()  # Set the model to inference mode
        else:
            self.model.train()  # Set the model to training mode

    def train_model(self, data: ModelDataset):
        # Example training loop
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(10):  # Example number of epochs
            running_loss = 0.0
            for example in data.examples:
                # Assuming each Example's features are already tensors
                inputs = torch.tensor([feature for feature in example.features.values()])
                labels = torch.tensor([1.0])  # Example label; replace with actual labels

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f'Epoch {epoch + 1}, Loss: {running_loss}')

    def run_inference(self, data: ModelDataset) -> ModelDataset:
        results = []
        with torch.no_grad():
            for example in data.examples:
                inputs = torch.tensor([feature for feature in example.features.values()])
                outputs = self.model(inputs)
                # Append results back into the dataset or create a new dataset
                results.append(outputs.item())  # Assuming single output

        # Optionally, return a new ModelDataset with the results
        return ModelDataset(examples=data.examples)
