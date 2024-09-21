'''
Utils for cGAN project.

Creted: 9/19/2024
Updated: 9/19/2024
'''
####################################################################################################################################
# Libraries
####################################################################################################################################

import os
import uuid
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from skimage.color import rgb2lab, lab2rgb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset



####################################################################################################################################
# Index
####################################################################################################################################

#1  set_requires_grad()  -> func
#2  lab_to_rgb()         -> func
#3  plot_batch()         -> func
#4  ModelInitializer     -> class
#5  PerceptualLoss       -> class
#6  ColorizationDataset  -> class
#7  Training             -> class
#8  Validation           -> class
#9  Visualization        -> class
#10 GANDriver            -> class
#11 select_images()      -> func

####################################################################################################################################
# Functions
####################################################################################################################################

####################################################################################################################################
# 1

def set_requires_grad(model, requires_grad=True):
    for p in model.parameters():
        p.requires_grad = requires_grad

####################################################################################################################################
# 2

def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).detach().cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis =0)

####################################################################################################################################
# 3

def plot_batch(fake_imgs, real_imgs, L):
    """
    Plot 4 images 
    """
    fig = plt.figure(figsize=(15, 8))
    for i in range(4):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    plt.show()

####################################################################################################################################
# 4

class ModelInitializer:
    def __init__(self, device, init_type='norm', gain=0.02):
        """
        Initializes a model on the specified device with a chosen weight initialization method.
        
        Args:
            device (torch.device): The device to move the model to (e.g., 'cuda' or 'cpu').
            init_type (str): The initialization method ('norm', 'xavier', 'kaiming'). Default is 'norm'.
            gain (float): The gain for initialization. Default is 0.02.
        """
        self.device = device
        self.init_type = init_type
        self.gain = gain
        

    def init_weights(self, net):
        """
        Initializes the weights of the network according to the specified method.
        
        Args:
            net (torch.nn.Module): The model whose weights will be initialized.
            
        Returns:
            torch.nn.Module: The model with initialized weights.
        """
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and 'Conv' in classname:
                if self.init_type == 'norm':
                    nn.init.normal_(m.weight.data, mean=0.0, std=self.gain)
                elif self.init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=self.gain)
                elif self.init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif 'BatchNorm2d' in classname:
                nn.init.normal_(m.weight.data, 1., self.gain)
                nn.init.constant_(m.bias.data, 0.0)

        net.apply(init_func)
        print(f"Model initialized with {self.init_type} initialization")
        return net

    def init_model(self, model, device):
        """
        Returns the initialized model.

        Args:
            model (torch.nn.Module): The model to be initialized.
        Returns:
            torch.nn.Module: The model initialized and moved to the specified device.
        """
        self.model = model.to(self.device)

        # Initialize weights of the model
        self.model = self.init_weights(self.model)

        return self.model


####################################################################################################################################
# 5

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

####################################################################################################################################
# 6

class ColorizationDataset(Dataset):
    def __init__(self, size, paths, split='train'):
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((size, size),  Image.BICUBIC),
                transforms.RandomHorizontalFlip(), # A little data augmentation!
            ])
        elif split == 'val':
            self.transforms = transforms.Resize((size, size),  Image.BICUBIC)
        
        self.split = split
        self.size = size
        self.paths = paths
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32") # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)
        L = (img_lab[[0], ...] / 50.) - 1. # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110. # Between -1 and 1
        
        return {'L': L, 'ab': ab}
    
    def __len__(self):
        return len(self.paths)
    

####################################################################################################################################
# 7

class Training:
    def __init__(self, generator, discriminator, optimizer_G, optimizer_D, adversarial_loss, l1_loss, lambda_l1, train_dl, device):
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.adversarial_loss = adversarial_loss
        self.l1_loss = l1_loss
        self.lambda_l1 = lambda_l1
        self.train_dl = train_dl
        self.device = device

    def train_epoch(self, epoch, epochs, show_fig=True):
        epoch_d_loss = 0
        epoch_g_loss = 0
        num_batches = 0
        train_loss_discriminator = []
        train_loss_generator = []

        # Use tqdm for the training loop
        pbar = tqdm(self.train_dl, desc=f"Training Epoch {epoch+1}/{epochs}")
        for i, data in enumerate(pbar):
            L, abs_ = data["L"], data["ab"]
            L, abs_ = L.to(self.device), abs_.to(self.device)
            batch_size = L.size(0)
            valid = torch.ones((batch_size, 1), requires_grad=False).to(self.device)
            fake = torch.zeros((batch_size, 1), requires_grad=False).to(self.device)

            # Generator
            generated_abs = self.generator(L)

            if show_fig:
                self.visualize_images(L, generated_abs, abs_)
                show_fig = False

            # Discriminator
            self.discriminator.train()
            for p in self.discriminator.parameters():
                p.requires_grad = True
            self.optimizer_D.zero_grad()

            fake_image = torch.cat([L, generated_abs.detach()], dim=1)
            fake_preds = self.discriminator(fake_image)
            real_images = torch.cat([L, abs_], dim=1)
            real_preds = self.discriminator(real_images)

            LOSS_D_FAKE = self.adversarial_loss(fake_preds, fake)
            LOSS_D_REAL = self.adversarial_loss(real_preds, valid)
            D_LOSS = (LOSS_D_FAKE + LOSS_D_REAL) / 2
            D_LOSS.backward()
            self.optimizer_D.step()

            # Generator
            self.generator.train()
            self.optimizer_G.zero_grad()
            for p in self.discriminator.parameters():
                p.requires_grad = False

            fake_image = torch.cat([L, generated_abs], dim=1)
            fake_preds = self.discriminator(fake_image)
            LOSS_G_GAN = self.adversarial_loss(fake_preds, valid)
            LOSS_L1 = self.l1_loss(generated_abs, abs_) * self.lambda_l1
            LOSS_G = LOSS_G_GAN + LOSS_L1
            LOSS_G.backward()
            self.optimizer_G.step()

            epoch_d_loss += D_LOSS.item()
            epoch_g_loss += LOSS_G.item()
            num_batches += 1

            # Update progress bar with current loss values
            pbar.set_postfix(D_loss=D_LOSS.item(), G_loss=LOSS_G.item())

        avg_d_loss = epoch_d_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches
        train_loss_discriminator.append(avg_d_loss)
        train_loss_generator.append(avg_g_loss)

        return train_loss_discriminator, train_loss_generator

    def visualize_images(self, L, generated_abs, abs_):
        L_plot = L[0:4, ...].clone()
        generated_abs_plot = generated_abs[0:4, ...].clone()
        abs_plot = abs_[0:4, ...]

        fake_images = lab_to_rgb(L_plot, generated_abs_plot)
        real_images = lab_to_rgb(L_plot, abs_plot)
        print("*" * 100)
        print("Generated Images -- Real Images -- Black and White")
        plot_batch(fake_images, real_images, L)

####################################################################################################################################
# 8

class Validation:
    def __init__(self, generator, discriminator, adversarial_loss, l1_loss, lambda_l1, val_dl, device):
        self.generator = generator
        self.discriminator = discriminator
        self.adversarial_loss = adversarial_loss
        self.l1_loss = l1_loss
        self.lambda_l1 = lambda_l1
        self.val_dl = val_dl
        self.device = device

    def validate_epoch(self, epoch, epochs):
        self.generator.eval()
        self.discriminator.eval()
        val_loss_discriminator = []
        val_loss_generator = []
        epoch_val_d_loss = 0
        epoch_val_g_loss = 0
        num_batches = 0

        # Use tqdm for the validation loop
        pbar = tqdm(self.val_dl, desc=f"Validation Epoch {epoch+1}/{epochs}")
        with torch.no_grad():
            for i, data in enumerate(pbar):
                L, abs_ = data["L"], data["ab"]
                L, abs_ = L.to(self.device), abs_.to(self.device)
                batch_size = L.size(0)
                real_labels = torch.ones((batch_size, 1), requires_grad=False).to(self.device)
                fake_labels = torch.zeros((batch_size, 1), requires_grad=False).to(self.device)

                # Generate ab values with generator
                generated_abs = self.generator(L)

                # Discriminator validation
                fake_image = torch.cat([L, generated_abs.detach()], dim=1)
                fake_preds = self.discriminator(fake_image)
                LOSS_D_FAKE = self.adversarial_loss(fake_preds, fake_labels)

                real_images = torch.cat([L, abs_], dim=1)
                real_preds = self.discriminator(real_images)
                LOSS_D_REAL = self.adversarial_loss(real_preds, real_labels)
                D_LOSS = (LOSS_D_FAKE + LOSS_D_REAL) / 2

                # Generator validation
                fake_image = torch.cat([L, generated_abs], dim=1)
                fake_preds = self.discriminator(fake_image)
                LOSS_G_GAN = self.adversarial_loss(fake_preds, real_labels)
                LOSS_L1 = self.l1_loss(generated_abs, abs_) * self.lambda_l1
                LOSS_G = LOSS_G_GAN + LOSS_L1

                epoch_val_d_loss += D_LOSS.item()
                epoch_val_g_loss += LOSS_G.item()
                num_batches += 1

                # Update progress bar with current loss values
                pbar.set_postfix(D_loss=D_LOSS.item(), G_loss=LOSS_G.item())

        avg_val_d_loss = epoch_val_d_loss / num_batches
        avg_val_g_loss = epoch_val_g_loss / num_batches
        val_loss_discriminator.append(avg_val_d_loss)
        val_loss_generator.append(avg_val_g_loss)

        return val_loss_discriminator, val_loss_generator

####################################################################################################################################
# 9

class Visualization:
    @staticmethod
    def plot_losses(epoch, train_loss_discriminator, val_loss_discriminator, train_loss_generator, val_loss_generator):
        epoch_range = range(1, epoch + 2)

        # Create a figure with 3 subplots, arranged horizontally
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Plot 1: Discriminator Train/Val Loss
        axs[0].plot(epoch_range, train_loss_discriminator, c="r", label="Train Loss")
        axs[0].plot(epoch_range, val_loss_discriminator, c="b", label="Val Loss")
        axs[0].set_title("Discriminator Train/Val Loss")
        axs[0].legend()

        # Plot 2: Generator Train/Val Loss
        axs[1].plot(epoch_range, train_loss_generator, c="r", label="Train Loss")
        axs[1].plot(epoch_range, val_loss_generator, c="b", label="Val Loss")
        axs[1].set_title("Generator Train/Val Loss")
        axs[1].legend()

        # Plot 3: Generator vs Discriminator Train Loss
        axs[2].plot(epoch_range, train_loss_discriminator, c="r", label="Discriminator Train Loss")
        axs[2].plot(epoch_range, train_loss_generator, c="g", label="Generator Train Loss")
        axs[2].set_title("Generator vs Discriminator Train Loss")
        axs[2].legend()

        # Display the plots
        plt.tight_layout()
        plt.show()

####################################################################################################################################
# 10

class GANDriver:
    def __init__(self, generator, discriminator, train_dl, val_dl, optimizer_G, optimizer_D, adversarial_loss, l1_loss, lambda_l1, device, epochs):
        self.device = device
        self.epochs = epochs
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.adversarial_loss = adversarial_loss
        self.l1_loss = l1_loss
        self.lambda_l1 = lambda_l1

        # Initialize the training and validation classes
        self.trainer = Training(self.generator, self.discriminator, self.optimizer_G, self.optimizer_D,
                                self.adversarial_loss, self.l1_loss, self.lambda_l1, train_dl, self.device)
        self.validator = Validation(self.generator, self.discriminator, self.adversarial_loss, 
                                    self.l1_loss, self.lambda_l1, val_dl, self.device)
        self.visualizer = Visualization()

        # Lists to store the training and validation losses
        self.train_loss_discriminator = []
        self.train_loss_generator = []
        self.val_loss_discriminator = []
        self.val_loss_generator = []

    def run(self):
        """
        Runs the training process and saves the model weights after training.

        """
        # Iterate over epochs
        for epoch in range(self.epochs):
            print(f"\n========== Epoch {epoch+1}/{self.epochs} ==========")

            # Perform training and validation for the current epoch
            train_d_loss, train_g_loss = self.trainer.train_epoch(epoch, self.epochs)
            val_d_loss, val_g_loss = self.validator.validate_epoch(epoch, self.epochs)

            # Append the losses to the corresponding lists
            self.train_loss_discriminator.extend(train_d_loss)
            self.train_loss_generator.extend(train_g_loss)
            self.val_loss_discriminator.extend(val_d_loss)
            self.val_loss_generator.extend(val_g_loss)

            # Visualize the losses
            self.visualizer.plot_losses(epoch, self.train_loss_discriminator, self.val_loss_discriminator,
                                        self.train_loss_generator, self.val_loss_generator)

        # Training complete, save the model weights
        self.save_model_weights()

        print("\nTraining complete and model weights saved.")

    def save_model_weights(self, base_dir='model_weights'):
        """
        Save the generator and discriminator weights to uniquely named subfolders.

        Args:
            base_dir (str): Base directory where model weights will be saved.
        """
        # Create a unique folder using timestamp and UUID
        unique_folder = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_' + str(uuid.uuid4())
        save_dir = os.path.join(base_dir, unique_folder)

        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Define paths for generator and discriminator weights
        save_path_generator = os.path.join(save_dir, 'generator_weights.pth')
        save_path_discriminator = os.path.join(save_dir, 'discriminator_weights.pth')

        # Save the model weights
        torch.save(self.generator.state_dict(), save_path_generator)
        torch.save(self.discriminator.state_dict(), save_path_discriminator)

        print(f"Generator weights saved to {save_path_generator}")
        print(f"Discriminator weights saved to {save_path_discriminator}")

####################################################################################################################################
# 11

def select_images(paths, num_images, split=0.8, seed=18):
    """
    Randomly select a subset of image paths and split them into training and validation sets.

    Args:
        paths (list): List of image paths to choose from.
        num_images (int): The total number of images to return (up to a maximum of 20000).
        split (float): Percent of values for train set. Default is 0.8
        seed (int): Random seed for reproducibility. Default is 18.

    Returns:
        tuple: Two lists, train_paths and val_paths.
    """
    if num_images > 20_000:
        raise ValueError("num_images cannot exceed 20000.")
    
    np.random.seed(seed)  # Set seed for reproducibility
    paths_subset = np.random.choice(paths, num_images, replace=False)  
    rand_idxs = np.random.permutation(num_images)  # Shuffle the indices

    # Split the data into training and validation sets
    train_size = int(split * num_images)  
    val_size = num_images - train_size  

    train_idxs = rand_idxs[:train_size]
    val_idxs = rand_idxs[train_size:train_size + val_size]

    # Select paths for train and validation
    train_paths = paths_subset[train_idxs]
    val_paths = paths_subset[val_idxs]

    return train_paths, val_paths


####################################################################################################################################
# END
####################################################################################################################################
if __name__ == "__Main__":
    pass