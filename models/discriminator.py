"""
Discriminator Networks
"""
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader

import numpy as np 
import os
import glob
import time
import numpy as np
#from PIL import Image
from pathlib import Path
#from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb


######################################################################################################################
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
    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()
        model = [self.get_layers(input_c, num_filters, norm=False)]
        model += [self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i + 1), s=1 if i == (n_down-1) else 2) 
                          for i in range(n_down)] # the 'if' statement is taking care of not using
                                                  # stride of 2 for the last block in this loop
        model += [self.get_layers(num_filters * 2 ** n_down, 1, s=1, norm=False, act=False)] # Make sure to not use normalization or
                                                                                             # activation for the last layer of the model
        self.model = nn.Sequential(*model)                                                   
        
    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True, act=True): # when needing to make some repeatitive blocks of layers,
        layers = [nn.Conv2d(ni, nf, k, s, p, bias=not norm)]          # it's always helpful to make a separate method for that purpose
        if norm: layers += [nn.BatchNorm2d(nf)]
        if act: layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

######################################################################################################################
class UnetBlock(nn.Module):
    def __init__(self, nf, ni, submodule=None, input_c=None, dropout=False,
                 innermost=False, outermost=False):
        super().__init__()
        self.outermost = outermost
        if input_c is None: input_c = nf
        downconv = nn.Conv2d(input_c, ni, kernel_size=4,
                             stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(ni)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(nf)
        
        if outermost:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4,
                                        stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(ni, nf, kernel_size=4,
                                        stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4,
                                        stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if dropout: up += [nn.Dropout(0.5)]
            model = down + [submodule] + up
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class Unet(nn.Module):
    def __init__(self, input_c=1, output_c=2, n_down=8, num_filters=64):
        super().__init__()
        unet_block = UnetBlock(num_filters * 8, num_filters * 8, innermost=True)
        for _ in range(n_down - 5):
            unet_block = UnetBlock(num_filters * 8, num_filters * 8, submodule=unet_block, dropout=True)
        out_filters = num_filters * 8
        for _ in range(3):
            unet_block = UnetBlock(out_filters // 2, out_filters, submodule=unet_block)
            out_filters //= 2
        self.model = UnetBlock(output_c, out_filters, input_c=input_c, submodule=unet_block, outermost=True)
    
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    pass

# Testing Outputs
discriminator = Discriminator_2()
dummy_input = torch.randn(4, 3, 256, 256)  # batch_size=4, channels=3, image_size=256x256
output = discriminator(dummy_input)
print("Expected output: (4, 1, 13, 13)")
print(f"Actual output: {output.shape}")
