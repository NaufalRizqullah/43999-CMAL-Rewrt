import torch
from torch import nn

from layers.basic_conv import *
from layers.features import *

class NetworkWrapper(nn.Module):
    def __init__(self, net_layers, num_classes):
        """
        Initializes a network wrapper module.

        Args:
            net_layers (list): List of network layers for feature extraction.
            num_classes (int): Number of classes for the classification task.
        """
        super().__init__()

        # Feature extraction module
        self.features = Features(net_layers)

        # Max pooling layers
        self.max_pool1 = nn.MaxPool2d(kernel_size=46, stride=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=23, stride=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=12, stride=1)

        # Convolutional block 1
        self.conv_block1 = nn.Sequential(
            BasicConv(512, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )

        # Classifier 1
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_classes)
        )

        # Convolutional block 2
        self.conv_block2 = nn.Sequential(
            BasicConv(1024, 512, kernel_size=1,
                      stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )

        # Classifier 2
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_classes)
        )

        # Convolutional block 3
        self.conv_block3 = nn.Sequential(
            BasicConv(2048, 512, kernel_size=1,
                      stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )

        # Classifier 3
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_classes)
        )

        # Combined classifier
        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(1024 * 3),
            nn.Linear(1024 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        Forward pass through the network wrapper.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Tuple of classification outputs and intermediate feature maps.
        """
        x1, x2, x3 = self.features(x)

        # Forward pass through convolutional block 1
        x1_ = self.conv_block1(x1)
        map1 = x1_.clone().detach()
        x1_ = self.max_pool1(x1_)
        x1_f = x1_.view(x1_.size(0), -1)
        x1_c = self.classifier1(x1_f)

        # Forward pass through convolutional block 2
        x2_ = self.conv_block2(x2)
        map2 = x2_.clone().detach()
        x2_ = self.max_pool2(x2_)
        x2_f = x2_.view(x2_.size(0), -1)
        x2_c = self.classifier2(x2_f)

        # Forward pass through convolutional block 3
        x3_ = self.conv_block3(x3)
        map3 = x3_.clone().detach()
        x3_ = self.max_pool3(x3_)
        x3_f = x3_.view(x3_.size(0), -1)
        x3_c = self.classifier2(x3_f)

        # Concatenate features from all blocks
        x_c = torch.cat([x1_f, x2_f, x3_f], -1)

        # Forward pass through the combined classifier
        x_c_all = self.classifier_concat(x_c)

        # Return classification outputs and intermediate feature maps
        return x1_c, x2_c, x3_c, x_c_all, map1, map2, map3
