from torch import nn

class Features(nn.Module):
    def __init__(self, net_layers_featurehead):
        super().__init__()

        self.net_layer_0 = nn.Sequential(net_layers_featurehead[0])
        self.net_layer_1 = nn.Sequential(*net_layers_featurehead[1])
        self.net_layer_2 = nn.Sequential(*net_layers_featurehead[2])
        self.net_layer_3 = nn.Sequential(*net_layers_featurehead[3])
        self.net_layer_4 = nn.Sequential(*net_layers_featurehead[4])
        self.net_layer_5 = nn.Sequential(*net_layers_featurehead[5])

    def forward(self, x):
        x = self.net_layer_0(x)
        x = self.net_layer_1(x)
        x = self.net_layer_2(x)

        x1 = self.net_layer_3(x)
        x2 = self.net_layer_4(x1)
        x3 = self.net_layer_5(x2)

        return x1, x2, x3