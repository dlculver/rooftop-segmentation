"""
We define a U-Net model with backbone a pre-trained MobileNet model
"""

import torch
from torch import nn  # pylint: disable=import-error

from torchvision.models import mobilenet_v2

from torchvision.models.feature_extraction import create_feature_extractor

# pylint: disable=invalid-name
# pylint: disable=no-member


import torch
import torch.nn as nn


class UpBlock(nn.Module):
    """
    Pytorch block for upsampling in convolutional neural networks

    Arguments:
        in_channels : int, number of channels for input
        out_channels : int, number of channels for output
        kernel_size : int or pair of ints, size of filters
        stride : int or pair of ints, stride of the filters
        padding : int or pair of ints, padding to be applied to input
        mode : str, defaults to 'bilinear'

    Returns:
        nn.Module class for upsampling in CNNs
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        padding=1,
        mode="bilinear",
    ):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=stride, mode=mode)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input_):
        x = self.upsample(input_)
        x = self.conv(x)
        x = self.batchnorm(x)
        output = self.relu(x)
        return output


class UNetMobileNet(nn.Module):
    """
    U-Net Model architecture

    Arguments:
        output_channels : int, the number of classes you wish to classify for semantic segmentation

    Returns:
        nn.Module class instance
    """

    def __init__(self, output_channels):
        super().__init__()
        base_model = mobilenet_v2(pretrained=True)

        # create feature extractor
        return_nodes = {
            "features.2.conv.0": "block2_expand_relu",
            "features.4.conv.0": "block4_expand_relu",
            "features.7.conv.0": "block7_expand_relu",
            "features.14.conv.0": "block14_expand_relu",
            "features.17.conv.2": "block17_project",
        }

        self.layer_names = [
            "block2_expand_relu",
            "block4_expand_relu",
            "block7_expand_relu",
            "block14_expand_relu",
            "block17_project",
        ]  # to be used in the forward pass

        self.down_stack = create_feature_extractor(
            base_model, return_nodes=return_nodes
        )

        self.up_stack = nn.ModuleList(
            [
                UpBlock(320, 512),
                UpBlock(512 + 576, 256),
                UpBlock(192 + 256, 128),
                UpBlock(144 + 128, 64),
            ]
        )

        self.last = nn.ConvTranspose2d(96 + 64, output_channels, 3, stride=2, padding=0)

    def forward(self, input_):
        """
        forward pass for module class
        """
        # grab extracted features from backbone
        skips = list(self.down_stack(input_).values())
        input_ = skips[-1]  # get deepest feature
        skips = reversed(skips[:-1])

        # upsampling and concatenating with skip connections
        for up, skip in zip(self.up_stack, skips):
            # print(f"shape before {up}: {input_.shape}")
            input_ = up(input_)
            # print(f"shape after {up}: {input_.shape}")
            # print(f"shape of skip: {skip.shape}")
            input_ = torch.cat((input_, skip), dim=1)
            # print(f"concatenated: {input_.shape}")

        # print(f"last up: {input_.shape}")
        out = self.last(input_)
        out = torch.sigmoid(out)

        return out
