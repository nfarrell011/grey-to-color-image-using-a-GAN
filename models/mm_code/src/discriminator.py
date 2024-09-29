import torch
from torch import nn, optim
import torch.nn.functional as F


 

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 4, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2)
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2)
        self.conv4 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 4, stride = 2)
        self.conv5 = nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 4, stride = 2)
        self.conv6 = nn.Conv2d(in_channels = 1024, out_channels = 1, kernel_size = 4, stride = 1)

        self.fc1 = nn.Linear(9, 1)

        self.batchnorm64 = nn.BatchNorm2d(64)
        self.batchnorm128 = nn.BatchNorm2d(128)
        self.batchnorm256 = nn.BatchNorm2d(256)
        self.batchnorm512 = nn.BatchNorm2d(512)
        self.batchnorm1024 = nn.BatchNorm2d(1024)

        self.leaky_relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = self.leaky_relu(self.batchnorm64(self.conv1(x)))
        x = self.leaky_relu(self.batchnorm128(self.conv2(x)))
        x = self.leaky_relu(self.batchnorm256(self.conv3(x)))
        x = self.leaky_relu(self.batchnorm512(self.conv4(x)))
        x = self.leaky_relu(self.batchnorm1024(self.conv5(x)))

        x = self.conv6(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)

        x = torch.sigmoid(x)

        return x
        
######################################################################################################################
class Discriminator_2(nn.Module):
    def __init__(self):
        super().__init__()

        # Conv layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=0)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=0)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=0)

        # Batch normalization layers
        self.batchnorm64 = nn.BatchNorm2d(64)
        self.batchnorm128 = nn.BatchNorm2d(128)
        self.batchnorm256 = nn.BatchNorm2d(256)
        self.batchnorm512 = nn.BatchNorm2d(512)


        # Leaky ReLU activation function
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        # Sigmoid function to produce a probability
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply conv, batch norm, and Leaky ReLU for each layer
        x = self.leaky_relu(self.batchnorm64(self.conv1(x)))
        x = self.leaky_relu(self.batchnorm128(self.conv2(x)))
        x = self.leaky_relu(self.batchnorm256(self.conv3(x)))
        x = self.leaky_relu(self.batchnorm512(self.conv4(x)))
        
        # Final convolution layer reduces to 1x1
        x = self.conv5(x)
        
        return x

######################################################################################################################
class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(PatchDiscriminator, self).__init__()
        
        # Define the sequential layers
        self.model = nn.Sequential(
            # Block 1: Input to 64 channels, no batch norm in the first block
            nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            # Block 2: 64 -> 128 channels
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            # Block 3: 128 -> 256 channels
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            # Block 4: 256 -> 512 channels
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            # Output layer: 512 -> 1 channel (patch-level decision)
            nn.Sequential(
                nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)  # No batch norm or activation here
            )
        )
    
    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    pass
