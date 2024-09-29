import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

class UnetBlock(nn.Module):
    def __init__(self, nf,          # number of output filters
                 ni,                # number of input filters
                 submodule=None,    # name of next bock int he chain
                 input_c=None,      # number of input channels
                 dropout=False,     # dropouts for regularization (noise)
                 innermost=False,   # True if innermost layer
                 outermost=False):  # True if outermost layer
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

class UNet(nn.Module):
    def __init__(self, input_c=1, output_c=2, n_down=8, num_filters=64):
        super().__init__()
        # Innermost block
        unet_block = UnetBlock(num_filters * 8, 
                               num_filters * 8, 
                               innermost=True)
        #Intermediate blocks
        for _ in range(n_down - 5):
            unet_block = UnetBlock(num_filters * 8, 
                                   num_filters * 8, 
                                   submodule=unet_block, 
                                   dropout=True)
        out_filters = num_filters * 8
        for _ in range(3):
            unet_block = UnetBlock(out_filters // 2, 
                                   out_filters, 
                                   submodule=unet_block)
            out_filters //= 2
        # Outermost block
        self.model = UnetBlock(output_c, 
                               out_filters, 
                               input_c=input_c, 
                               submodule=unet_block, 
                               outermost=True)

    def forward(self, x):
        return self.model(x)
from torchvision.models import ResNet34_Weights, ResNet50_Weights

class UNetResNetGenerator(nn.Module):
    def __init__(self, input_channels=1, output_channels=2, base_model="resnet34", pretrained=True):
        super(UNetResNetGenerator, self).__init__()

        # Load the pretrained resnet backbone
        if base_model == "resnet34":
            self.encoder = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        elif base_model == "resnet50":
            self.encoder = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            raise ValueError("Base model must be 'resnet34' or 'resnet50'")

        # Modify the first convolution layer to accept 1 input channel
        self.encoder.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Create the decoder part (upsampling path)
        self.up1 = self.up_block(512, 256)  # ResNet Bottleneck Output (8x8 -> 16x16)
        self.up2 = self.up_block(256, 128)  # (16x16 -> 32x32)
        self.up3 = self.up_block(128, 64)   # (32x32 -> 64x64)
        self.up4 = self.up_block(64, 64)    # (64x64 -> 128x128)

        # Add an additional upsampling block to match dimensions for concatenation
        self.up5 = self.up_block(64, 64)    # (128x128 -> 256x256)

        # Final convolution layer to output 2 channels
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def up_block(self, in_c, out_c):
        """
        Create an upsampling block with ConvTranspose2d and BatchNorm.
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder (ResNet feature extraction path)
        x1 = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(x)))  # First layer
        x1 = self.encoder.maxpool(x1)

        x2 = self.encoder.layer1(x1)  # 64 feature maps
        x3 = self.encoder.layer2(x2)  # 128 feature maps
        x4 = self.encoder.layer3(x3)  # 256 feature maps
        x5 = self.encoder.layer4(x4)  # 512 feature maps

        # Decoder (upsampling path)
        u1 = self.up1(x5)              # 8x8 -> 16x16
        u2 = self.up2(u1 + x4)         # 16x16 -> 32x32
        u3 = self.up3(u2 + x3)         # 32x32 -> 64x64
        u4 = self.up4(u3 + x2)         # 64x64 -> 128x128
        u5 = self.up5(u4)              # 128x128 -> 256x256  (match dimensions for concatenation)

        # Final convolution
        out = self.final_conv(u5)
        return torch.tanh(out)

class UNetGenerator(nn.Module):
    def __init__(self, input_channels=1, output_channels=2, num_filters=64):
        super(UNetGenerator, self).__init__()

        # Downsampling path (Encoder)
        self.down1 = self.conv_block(input_channels, num_filters)      # 1 -> 64
        self.down2 = self.conv_block(num_filters, num_filters * 2)     # 64 -> 128
        self.down3 = self.conv_block(num_filters * 2, num_filters * 4) # 128 -> 256
        self.down4 = self.conv_block(num_filters * 4, num_filters * 8) # 256 -> 512
        self.down5 = self.conv_block(num_filters * 8, num_filters * 8) # 512 -> 512 (bottleneck)

        # Upsampling path (Decoder)
        self.up1 = self.up_block(num_filters * 8, num_filters * 8)     # 512 -> 512
        self.up2 = self.up_block(num_filters * 16, num_filters * 4)    # 1024 -> 256
        self.up3 = self.up_block(num_filters * 8, num_filters * 2)     # 512 -> 128
        self.up4 = self.up_block(num_filters * 4, num_filters)         # 256 -> 64
        self.up5 = self.up_block(num_filters * 2, num_filters)         # New block: 128 -> 64 to recover full resolution

        # Final output layer (1x1 Conv to get the ab channels)
        self.final_conv = nn.Conv2d(num_filters, output_channels, kernel_size=1)

    def conv_block(self, in_c, out_c):
        """
        Create two convolutional layers followed by BatchNorm and LeakyReLU activation.
        The first convolution uses a stride of 2 to downsample instead of MaxPooling.
        """
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1),  # Stride 2 for downsampling
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def up_block(self, in_c, out_c):
        """
        Create an upsampling block with ConvTranspose2d and concatenation from the corresponding downsampling layer.
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Downsampling with skip connections
        d1 = self.down1(x)  # 64x128x128
        d2 = self.down2(d1)  # 128x64x64
        d3 = self.down3(d2)  # 256x32x32
        d4 = self.down4(d3)  # 512x16x16
        d5 = self.down5(d4)  # 512x8x8 (bottleneck)

        # Upsampling with skip connections
        u1 = self.up1(d5)
        u1 = torch.cat([u1, d4], dim=1)  # Concatenate skip connection from d4 (1024x16x16)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d3], dim=1)  # Concatenate skip connection from d3 (512x32x32)
        u3 = self.up3(u2)
        u3 = torch.cat([u3, d2], dim=1)  # Concatenate skip connection from d2 (256x64x64)
        u4 = self.up4(u3)
        u4 = torch.cat([u4, d1], dim=1)  # Concatenate skip connection from d1 (128x128x128)
        u5 = self.up5(u4)  # New block to upsample back to 256x256

        # Final output layer
        return torch.tanh(self.final_conv(u5))  # Output 2x256x256 (ab channels)
    
if __name__ == "__main__":
    pass