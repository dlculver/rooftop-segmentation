"""
We define a U-Net model with backbone a pre-trained MobileNet model
"""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v2
from torchvision.models.feature_extraction import create_feature_extractor

from tqdm import tqdm


def train_model(
        model, 
        device, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer,
        scheduler, 
        num_epochs):
    
    train_losses, val_losses = [], []

    model.to(device)

    for epoch in tqdm(range(num_epochs), ):

        model.train()
        running_loss = 0.
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(images)

             # Crop the model's output to match the size of the labels (256x256)
            labels = F.interpolate(labels, size=outputs.size()[2:], mode='nearest') # more methodical way of making label and outputs same size

            # Compute the loss
            loss = criterion(outputs, labels)

            # Backpropagation and optimizer step
            loss.backward()
            optimizer.step()

            # update running loss
            running_loss += loss.item() * images.size(0)

        # calculate average loss over the epoch
        average_train_loss = running_loss / len(train_loader)
        train_losses.append(average_train_loss)

        # evaluation on validation set
        model.eval()
        running_val_loss = 0.

        with torch.no_grad():
            for image, label in val_loader:
                image, label = image.to(device), label.to(device)
                output = model(image)
                
                label = F.interpolate(label, size=output.size()[2:], mode='nearest')

                loss = criterion(output, label)

                running_val_loss += loss.item()

                # step scheduler
                if scheduler is not None:
                    scheduler.step()   

        average_val_loss = running_val_loss / len(val_loader)
        val_losses.append(average_val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}] Average Train Loss: {average_train_loss:.4f}, Average Val Loss: {average_val_loss:.4f}")

    return model, train_losses, val_losses                

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
        dropout_rate: float = None,
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
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate else None

    def forward(self, input_):
        x = self.upsample(input_)
        x = self.conv(x)
        x = self.batchnorm(x)
        output = self.relu(x)
        if self.dropout is not None:
            output = self.dropout(output)
        return output


class UNetMobileNet(nn.Module):
    """
    U-Net Model architecture

    Arguments:
        output_channels : int, the number of classes you wish to classify for semantic segmentation

    Returns:
        nn.Module class instance
    """

    def __init__(self, output_channels, dropout_rate = None):
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
                UpBlock(320, 512, dropout_rate=dropout_rate),
                UpBlock(512 + 576, 256, dropout_rate=dropout_rate),
                UpBlock(192 + 256, 128, dropout_rate=dropout_rate),
                UpBlock(144 + 128, 64, dropout_rate=dropout_rate),
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
    

