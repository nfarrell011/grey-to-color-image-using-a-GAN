import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

class PerceptualLoss(nn.Module):
    def __init__(self, model='vgg16', layer=2):
        """
        Perceptual loss based on feature maps extracted from a pretrained VGG16 model.
        
        Args:
            model (str): Select vgg16 or vgg19
            layer (int): The layer index from which to extract feature maps for the loss.
                         Layers can range from 0 (first convolutional layer) to the final layer.
        """
        super(PerceptualLoss, self).__init__()

        # Load a pretrained model on ImageNet
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dict = {'vgg16': models.vgg16, 'vgg19': models.vgg19}
        self.model = model_dict[model](pretrained=True).features
        self.model.eval()                                                       # Keep the VGG16 model in evaluation mode (no gradients needed)
        self.model.requires_grad_(False)                                        # Ensure the VGG model is not trainable
        self.model.to(self.device)
        self.layer = layer                                                      # Layer to extract feature maps from (e.g., 9 for features after 3rd convolution block)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],       # Nomalization for input iamges to match ImageNet norm for best use with pretrained VGG
                                              std=[0.229, 0.224, 0.225])
        self.loss_func = F.mse_loss                                             # L2 Loss Function
    
    def forward(self, real_images, fake_images):
        """
        Computes the perceptual loss (L2 loss between feature maps) for the real and generated images.
        
        Args:
            real_image (torch.Tensor): The ground truth image (real image) with shape [B, C, H, W].
            generated_image (torch.Tensor): The generated image with shape [B, C, H, W].
        
        Returns:
            torch.Tensor: The perceptual loss between real and generated images.
        """
        # Ensure both images are normalized in the same way VGG16 expects (e.g., ImageNet stats)
        real_features = self.get_features(real_images.to(self.device))
        fake_features = self.get_features(fake_images.to(self.device))
        
        # Calculate L2 loss (mean squared error) between feature maps
        loss = self.loss_func(fake_features, real_features)
        return loss

    def get_features(self, x):
        """
        Extracts the feature maps from the VGG16 model up to the specified layer.
        
        Args:
            x (torch.Tensor): The input image tensor.
        
        Returns:
            torch.Tensor: The extracted feature maps.
        """
        features = self.normalize(x)
        for i, layer in enumerate(self.model):
            features = layer(features)
            if i == self.layer:
                break
        return features


if __name__ == "__main__":
    # Test usage

    # Random input of shape [B, C, H, W]
    real_image = torch.randn(4, 3, 256, 256) 
    generated_image = torch.randn(4, 3, 256, 256)  

    # Initialize
    perceptual_loss = PerceptualLoss(layer=9)
    loss = perceptual_loss(real_image, generated_image)
    print(f"Perceptual Loss: {loss}")