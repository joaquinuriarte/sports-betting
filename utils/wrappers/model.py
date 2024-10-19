import torch.nn as nn


class Model(nn.Module):
    """
    A wrapper for constructing a neural network model dynamically from a configuration.

    This class builds the model architecture using the provided configuration which defines
    layers such as Linear and activation functions like ReLU.

    Attributes:
        layers (nn.Sequential): A sequential container that holds the layers of the model.
    """

    def __init__(self, architecture_config: dict):
        super(Model, self).__init__()

        layers = []

        # Loop through each layer configuration in the architecture config
        for layer_config in architecture_config["layers"]:
            if layer_config["type"] == "Linear":
                layers.append(
                    nn.Linear(layer_config["in_features"], layer_config["out_features"])
                )
            elif layer_config["type"] == "ReLU":
                layers.append(nn.ReLU())
            # You can add more layer types here if needed, like Dropout, BatchNorm, etc.

        # Stack the layers together using nn.Sequential
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Defines the forward pass of the model. The input `x` passes through all the layers.

        Args:
            x: Input tensor.

        Returns:
            Output after passing through the model's layers.
        """
        return self.layers(x)
