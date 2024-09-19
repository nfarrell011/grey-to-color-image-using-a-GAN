import torch
from torch import nn, optim
import torch.nn.functional as F

import torch
from torch import nn, optim
from torch.nn import nn
import torch.nn.functional as F
import numpy as np 
import pandas as pd
import cv2  

class Descriminator(nn.Module):
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
        print(f"This is the x after the Conv6: {x}")

        x = torch.flatten(x, 1)
        print(f"X after flatten {x}")

        x = self.fc1(x)
        print(f"X after linear layer: {x}")

        x = torch.sigmoid(x)
        print(f"After Sig: {x}")

        return x
        
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, gen_inputs, true_targets, descrim_out):
        loss = ((true_targets - gen_inputs) ** 2).mean() + 10e-3 * -1 * descrim_out
if __name__ == "__main__":
    pass
