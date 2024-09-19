import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Downsampling layers
        self.down1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.down2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.down3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.down4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.down5 = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1)

        # Upsampling layers
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.up5 = nn.ConvTranspose2d(64, 2, kernel_size=4, stride=2, padding=1)

        # Activation functions and batch norm
        self.relu = nn.ReLU(True)
        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.batchnorm64 = nn.BatchNorm2d(64)
        self.batchnorm128 = nn.BatchNorm2d(128)
        self.batchnorm256 = nn.BatchNorm2d(256)
        self.batchnorm512 = nn.BatchNorm2d(512)

    def forward(self, x):
        # Downsample
        d1 = self.leaky_relu(self.batchnorm64(self.down1(x)))
        d2 = self.leaky_relu(self.batchnorm128(self.down2(d1)))
        d3 = self.leaky_relu(self.batchnorm256(self.down3(d2)))
        d4 = self.leaky_relu(self.batchnorm512(self.down4(d3)))
        d5 = self.leaky_relu(self.down5(d4))  # bottleneck

        # Upsample
        u1 = self.relu(self.up1(d5))
        u2 = self.relu(self.batchnorm256(self.up2(u1 + d4)))  # skip connection
        u3 = self.relu(self.batchnorm128(self.up3(u2 + d3)))  # skip connection
        u4 = self.relu(self.batchnorm64(self.up4(u3 + d2)))  # skip connection
        u5 = torch.tanh(self.up5(u4 + d1))  # skip connection

        return u5
    
if __name__ == "__main__":
    pass