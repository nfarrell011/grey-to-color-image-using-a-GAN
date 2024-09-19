import torch
import torch.nn as nn

class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(PatchDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),  # (N, 64, H/2, W/2)
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (N, 128, H/4, W/4)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # (N, 256, H/8, W/8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # (N, 512, H/16, W/16)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),  # (N, 1, H/16 - 3, W/16 - 3)
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":

    # Testing  PatchDiscriminator
    discriminator = PatchDiscriminator()
    dummy_input = torch.randn(4, 3, 256, 256)  # batch_size=4, channels=3, image_size=256x256
    output = discriminator(dummy_input)
    print("Expected output: (4, 1, 13, 13)")
    print(f"Actual output: {output.shape}")