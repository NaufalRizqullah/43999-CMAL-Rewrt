from torch import nn

class BasicConv(nn.Module):
    def __init__(
            self,
            in_planes,
            out_planes,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            relu=True,
            bn=True,
            bias=False
    ):
        """
        Initializes a basic convolutional module.

        Args:
            in_planes (int): Number of input channels.
            out_planes (int): Number of output channels.
            kernel_size (int or tuple): Size of the convolutional kernel.
            stride (int, optional): Stride of the convolution. Defaults to 1.
            padding (int, optional): Padding added to input. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            groups (int, optional): Number of blocked connections. Defaults to 1.
            relu (bool, optional): Whether to apply ReLU activation. Defaults to True.
            bn (bool, optional): Whether to apply batch normalization. Defaults to True.
            bias (bool, optional): Whether to include bias in the convolution. Defaults to False.
        """
        super().__init__()

        self.out_channels = out_planes

        # Conv Layer
        self.conv = nn.Conv2d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

        # Batch Normalization Layer
        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=1e-5,
            momentum=0.01,
            affine=True
        ) if bn else None

        # ReLU Layer
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        """
        Forward pass through the basic convolutional module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv(x)

        if self.bn is not None:
            x = self.bn(x)

        if self.relu is not None:
            x = self.relu(x)

        return x