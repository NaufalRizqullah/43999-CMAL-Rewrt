from torch import nn

class Features(nn.Module):
    def __init__(self, net_layers_featurehead):
        """
        Initializes a features extraction module.

        Args:
            net_layers_featurehead (list): List of network layers for feature extraction.
        """
        super().__init__()

        # Define network layers for feature extraction
        self.net_layer_0 = nn.Sequential(net_layers_featurehead[0])
        self.net_layer_1 = nn.Sequential(*net_layers_featurehead[1])
        self.net_layer_2 = nn.Sequential(*net_layers_featurehead[2])
        self.net_layer_3 = nn.Sequential(*net_layers_featurehead[3])
        self.net_layer_4 = nn.Sequential(*net_layers_featurehead[4])
        self.net_layer_5 = nn.Sequential(*net_layers_featurehead[5])

    def forward(self, x):
        """
        Forward pass through the features extraction module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Tuple of feature tensors from different layers.
        """
        # Apply each network layer sequentially
        x = self.net_layer_0(x)
        x = self.net_layer_1(x)
        x = self.net_layer_2(x)

        # Extract features from different layers
        x1 = self.net_layer_3(x)
        x2 = self.net_layer_4(x1)
        x3 = self.net_layer_5(x2)

        return x1, x2, x3