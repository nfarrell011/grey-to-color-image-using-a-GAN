import torch
import torch.nn as nn

# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet, self).__init__()

#         # Downsampling layers
#         self.down1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
#         self.down2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
#         self.down3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
#         self.down4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
#         self.down5 = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1)

#         # Upsampling layers
#         self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
#         self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
#         self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
#         self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
#         self.up5 = nn.ConvTranspose2d(64, 2, kernel_size=4, stride=2, padding=1)

#         # Activation functions and batch norm
#         self.relu = nn.ReLU(True)
#         self.leaky_relu = nn.LeakyReLU(0.2, True)
#         self.batchnorm64 = nn.BatchNorm2d(64)
#         self.batchnorm128 = nn.BatchNorm2d(128)
#         self.batchnorm256 = nn.BatchNorm2d(256)
#         self.batchnorm512 = nn.BatchNorm2d(512)

#     def forward(self, x):
#         # Downsample
#         d1 = self.leaky_relu(self.batchnorm64(self.down1(x)))
#         d2 = self.leaky_relu(self.batchnorm128(self.down2(d1)))
#         d3 = self.leaky_relu(self.batchnorm256(self.down3(d2)))
#         d4 = self.leaky_relu(self.batchnorm512(self.down4(d3)))
#         d5 = self.leaky_relu(self.down5(d4))  # bottleneck

#         # Upsample
#         u1 = self.relu(self.up1(d5))
#         u2 = self.relu(self.batchnorm256(self.up2(u1 + d4)))  # skip connection
#         u3 = self.relu(self.batchnorm128(self.up3(u2 + d3)))  # skip connection
#         u4 = self.relu(self.batchnorm64(self.up4(u3 + d2)))  # skip connection
#         u5 = torch.tanh(self.up5(u4 + d1))  # skip connection

#         return u5



class UNet(nn.Module):
    def __init__(self, input_c=1, output_c=2, num_filters=64):
        super(UNet, self).__init__()

        # Downsampling path (Encoder)
        self.down1 = self.conv_block(input_c, num_filters)          # 1 -> 64
        self.down2 = self.conv_block(num_filters, num_filters * 2)  # 64 -> 128
        self.down3 = self.conv_block(num_filters * 2, num_filters * 4)  # 128 -> 256
        self.down4 = self.conv_block(num_filters * 4, num_filters * 8)  # 256 -> 512
        self.down5 = self.conv_block(num_filters * 8, num_filters * 8)  # 512 -> 512 (bottleneck)

        # Upsampling path (Decoder)
        self.up1 = self.up_block(num_filters * 16, num_filters * 8)  # 1024 -> 512
        self.up2 = self.up_block(num_filters * 16, num_filters * 4)  # 512 + 512 (skip) -> 256
        self.up3 = self.up_block(num_filters * 8, num_filters * 2)   # 256 + 256 (skip) -> 128
        self.up4 = self.up_block(num_filters * 4, num_filters)       # 128 + 128 (skip) -> 64

        # Final output layer
        self.final_conv = nn.Conv2d(num_filters, output_c, kernel_size=1)

    def conv_block(self, in_c, out_c):
        """
        Create two convolutional layers with kernel size 4 followed by BatchNorm and LeakyReLU activation.
        The first convolution uses a stride of 2 to downsample instead of MaxPooling.
        """
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1),  # Stride 2 for downsampling with kernel size 4
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=4, padding=1),  # Convolution with kernel size 4
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def up_block(self, in_c, out_c):
        """
        Create an upsampling block with ConvTranspose2d with kernel size 4 and concatenation from downsampling.
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1),  # Transpose conv with kernel size 4
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=4, padding=1),  # Convolution with kernel size 4
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=4, padding=1),  # Convolution with kernel size 4
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Downsampling with skip connections (using stride 2 convolutions for downsampling)
        d1 = self.down1(x)  # Downsample 1st block
        d2 = self.down2(d1)  # Downsample 2nd block
        d3 = self.down3(d2)  # Downsample 3rd block
        d4 = self.down4(d3)  # Downsample 4th block
        d5 = self.down5(d4)  # Bottleneck (innermost layer)

        # Upsampling with skip connections
        u1 = self.up1(d5)
        u1 = torch.cat([u1, d4], dim=1)  # Concatenate skip connection from down4
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d3], dim=1)  # Concatenate skip connection from down3
        u3 = self.up3(u2)
        u3 = torch.cat([u3, d2], dim=1)  # Concatenate skip connection from down2
        u4 = self.up4(u3)
        u4 = torch.cat([u4, d1], dim=1)  # Concatenate skip connection from down1

        # Final output layer
        return torch.tanh(self.final_conv(u4))
    
if __name__ == "__main__":
    pass