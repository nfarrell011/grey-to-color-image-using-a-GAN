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
import logging
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
#7  visualize_images     -> func
#8  Training             -> class
#9  Validation           -> class
#10  Visualization       -> class
#11 GANDriver            -> class
#12 select_images()      -> func
#13 EntropyLoss          -> class

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

import matplotlib.pyplot as plt

def plot_batch(fake_imgs, real_imgs, L, show=True, save_path=None):
    """
    Plot or save 4 images.

    Args:
        fake_imgs (list): List of generated images.
        real_imgs (list): List of real images.
        L (list): List of black and white images (L channel).
        show (bool): If True, display the images. Defaults to True.
        save_path (str): Path to save the images. If None, the images are not saved. Defaults to None.
    """
    fig = plt.figure(figsize=(15, 8))

    for i in range(4):
        # Plot the L channel (black and white)
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")

        # Plot the generated (fake) images
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i])
        ax.axis("off")

        # Plot the real images
        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")

    # Show the plot if requested
    if show:
        plt.show()
    
    # Save the plot if a save path is provided
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()  # Close the plot to free up memory

####################################################################################################################################
# 4

class ModelInitializer:
    def __init__(self, device, init_type='norm', gain=0.02):
        self.device = device
        self.init_type = init_type
        self.gain = gain
        # Set up a logger specific to this class
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Created with init_type={self.init_type}, gain={self.gain}")

    def init_weights(self, net):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and 'Conv' in classname:
                if self.init_type == 'norm':
                    nn.init.normal_(m.weight.data, mean=0.0, std=self.gain)
                    self.logger.info(f"Initialized {classname} with normal distribution (mean=0.0, std={self.gain})")
                elif self.init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=self.gain)
                    self.logger.info(f"Initialized {classname} with Xavier normal distribution (gain={self.gain})")
                elif self.init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                    self.logger.info(f"Initialized {classname} with Kaiming normal distribution")
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
                    self.logger.info(f"Initialized bias for {classname} with constant 0.0")
            elif 'BatchNorm2d' in classname:
                nn.init.normal_(m.weight.data, 1.0, self.gain)
                nn.init.constant_(m.bias.data, 0.0)
                self.logger.info(f"Initialized {classname} with normal distribution for weight (mean=1.0, std={self.gain}) and constant bias 0.0")

        net.apply(init_func)
        self.logger.info(f"Model initialized with {self.init_type} initialization")
        return net

    def init_model(self, model):
        self.model = model.to(self.device)
        self.logger.info(f"Model moved to {self.device}")
        self.model = self.init_weights(self.model)
        return self.model


####################################################################################################################################
# 5

class PerceptualLoss(nn.Module):
    def __init__(self, layer=9):
        """
        Perceptual loss based on feature maps extracted from a pretrained VGG19 model.
        
        Args:
            layer (int): The layer index from which to extract feature maps for the loss.
                         Layers can range from 0 (first convolutional layer) to the final layer.
        """
        super(PerceptualLoss, self).__init__()

        # Set up a logger specific to this class
        self.logger = logging.getLogger(self.__class__.__name__)

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"PerceptualLoss initialized on device: {self.device}")

        # Load a pretrained VGG19 model
        self.model = models.vgg19(pretrained=True).features
        self.model.eval()  # Keep the VGG model in evaluation mode
        self.model.requires_grad_(False)  # Ensure the VGG model is not trainable
        self.model.to(self.device)
        self.layer = layer  # Layer to extract feature maps from (e.g., 9 for features after 3rd convolution block)
        self.logger.info(f"VGG19 model loaded, using layer {self.layer} for feature extraction")

        # Normalization for input images to match ImageNet statistics
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.loss_func = F.mse_loss  # L2 Loss Function
    
    def forward(self, real_images, fake_images):
        """
        Computes the perceptual loss (L2 loss between feature maps) for the real and generated images.
        
        Args:
            real_image (torch.Tensor): The ground truth image (real image) with shape [B, C, H, W].
            generated_image (torch.Tensor): The generated image with shape [B, C, H, W].
        
        Returns:
            torch.Tensor: The perceptual loss between real and generated images.
        """
        self.logger.info(f"Calculating perceptual loss for {real_images.shape[0]} images")

        # Normalize and extract features for both real and fake images
        real_features = self.get_features(real_images.to(self.device))
        fake_features = self.get_features(fake_images.to(self.device))
        
        # Calculate L2 loss (mean squared error) between feature maps
        loss = self.loss_func(fake_features, real_features)
        self.logger.info(f"Perceptual loss calculated: {loss.item()}")
        return loss

    def get_features(self, x):
        """
        Extracts the feature maps from the VGG19 model up to the specified layer.
        
        Args:
            x (torch.Tensor): The input image tensor.
        
        Returns:
            torch.Tensor: The extracted feature maps.
        """
        self.logger.info(f"Extracting features from input tensor of shape {x.shape}")
        
        features = self.normalize(x)
        for i, layer in enumerate(self.model):
            features = layer(features)
            if i == self.layer:
                self.logger.info(f"Reached target layer {self.layer}, stopping feature extraction")
                break
        return features

####################################################################################################################################
# 6

class ColorizationDataset(Dataset):
    def __init__(self, size, paths, split='train'):
        """
        Initializes the ColorizationDataset with a specific image size and path list.
        
        Args:
            size (int): The target size to resize the images to (width and height).
            paths (list): List of file paths to the images.
            split (str): Whether the dataset is for 'train' or 'val'. Default is 'train'.
        """
        # Set up a logger specific to this class
        self.logger = logging.getLogger(self.__class__.__name__)

        # Data transformations depending on whether it's a training or validation set
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((size, size), Image.BICUBIC),
                transforms.RandomHorizontalFlip(),  # A little data augmentation
            ])
            self.logger.info(f"Training dataset initialized with size {size} and random horizontal flipping")
        elif split == 'val':
            self.transforms = transforms.Resize((size, size), Image.BICUBIC)
            self.logger.info(f"Validation dataset initialized with size {size}")

        self.split = split
        self.size = size
        self.paths = paths
        self.logger.info(f"Dataset created with {len(self.paths)} images")

    def __getitem__(self, idx):
        """
        Loads and transforms an image at the specified index.
        
        Args:
            idx (int): Index of the image to load.
        
        Returns:
            dict: A dictionary containing the L channel and ab channels of the image.
        """
        img_path = self.paths[idx]
        self.logger.info(f"Loading image: {img_path}")

        # Load image and apply transformations
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32")  # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)

        # Normalize L and ab channels to [-1, 1]
        L = (img_lab[[0], ...] / 50.) - 1.  # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110.    # Between -1 and 1
        
        self.logger.info(f"Image {img_path} transformed to L*a*b and normalized")
        
        return {'L': L, 'ab': ab}

    def __len__(self):
        """
        Returns the total number of images in the dataset.
        
        Returns:
            int: Number of images in the dataset.
        """
        return len(self.paths)
        
####################################################################################################################################  
# 7
def visualize_images(L, generated_abs, abs_, epoch, batch_idx, image_save_dir, show_fig=False, save_images=True):
    """
    Visualizes and/or saves a batch of generated images and their corresponding ground truth images.
    
    Args:
        L (torch.Tensor): The L channel of the input images.
        generated_abs (torch.Tensor): The generated ab channels from the generator.
        abs_ (torch.Tensor): The real ab channels.
        epoch (int): The current epoch number.
        batch_idx (int): The index of the current batch.
        show_fig (bool): Whether to display the images.
        save_images (bool): Whether to save the images.
    """
    L_plot = L[0:4, ...].clone()
    generated_abs_plot = generated_abs[0:4, ...].clone()
    abs_plot = abs_[0:4, ...]

    fake_images = lab_to_rgb(L_plot, generated_abs_plot)
    real_images = lab_to_rgb(L_plot, abs_plot)

    if show_fig:
        print("*" * 100)
        print("Generated Images -- Real Images -- Black and White")
        plot_batch(fake_images, real_images, L)
        # self.logger.info("Images visualized and plotted")

    if save_images:
        # Save images as a batch in the 'training_images' directory
        image_filename = os.path.join(image_save_dir, f'epoch_{epoch+1}_batch_{batch_idx+1}.png')
        # self.logger.info(f"Saving images to {image_filename}")
        plot_batch(fake_images, real_images, L, show=False, save_path=image_filename)
        
####################################################################################################################################
# 8

class Training:
    def __init__(self, generator, discriminator, optimizer_G, optimizer_D, adversarial_loss, content_loss, lambda_l1, train_dl, device, scheduler_D, scheduler_G, run_dir, base_dir='training_runs'):
        """
        Initializes the Training class with the models, optimizers, loss functions, data loaders, and device configuration.
        
        Args:
            run_dir (str): The specific directory for the current run (e.g., 'run_1').
            base_dir (str): The base directory where all runs are stored (default: 'training_runs').
        """
        # Set up a logger specific to this class
        self.logger = logging.getLogger(self.__class__.__name__)

        self.generator = generator
        self.discriminator = discriminator
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.adversarial_loss = adversarial_loss
        self.content_loss = content_loss
        self.lambda_l1 = lambda_l1
        self.train_dl = train_dl
        self.device = device
        self.scheduler_G = scheduler_G
        self.scheduler_D = scheduler_D

        # Directory for saving images
        self.image_save_dir = os.path.join(base_dir, run_dir, 'training_images')
        os.makedirs(self.image_save_dir, exist_ok=True)

        self.logger.info(f"Training initialized on device: {self.device} with images saved in {self.image_save_dir}")

    def train_epoch(self, epoch, epochs, show_fig=True, save_images=True):
        """
        Trains the models for one epoch and calculates the average discriminator and generator losses.
        
        Args:
            epoch (int): The current epoch number.
            epochs (int): The total number of epochs to train.
            show_fig (bool): Whether to visualize images during training.
            save_images (bool): Whether to save the generated and real images during training.
        
        Returns:
            tuple: Lists of average discriminator and generator losses for the epoch.
        """
        self.logger.info(f"Starting epoch {epoch+1}/{epochs}")

        epoch_d_loss = 0
        epoch_g_loss = 0
        num_batches = 0
        train_loss_discriminator = []
        train_loss_generator = []

        # Use tqdm for the training loop with a progress bar
        pbar = tqdm(self.train_dl, desc=f"Training Epoch {epoch+1}/{epochs}")
        for i, data in enumerate(pbar):
            L, abs_ = data["L"], data["ab"]
            L, abs_ = L.to(self.device), abs_.to(self.device)
            batch_size = L.size(0)
            valid = torch.ones((batch_size, 1), requires_grad=False).to(self.device)
            fake = torch.zeros((batch_size, 1), requires_grad=False).to(self.device)
            valid_d = torch.full((batch_size, 1), 0.9, device=self.device) 
            fake_d = torch.full((batch_size, 1), 0.1, device=self.device)
            
            # Generator forward pass
            generated_abs = self.generator(L)

            if show_fig or save_images:
                visualize_images(L, generated_abs, abs_, epoch, i, self.image_save_dir)
                show_fig = False
                save_images = False

            # Train the discriminator
            self.discriminator.train()
            for p in self.discriminator.parameters():
                p.requires_grad = True
            self.optimizer_D.zero_grad()

            fake_image = torch.cat([L, generated_abs.detach()], dim=1)
            fake_preds = self.discriminator(fake_image)
            real_images = torch.cat([L, abs_], dim=1)
            real_preds = self.discriminator(real_images)

            LOSS_D_FAKE = self.adversarial_loss(fake_preds, fake_d)
            LOSS_D_REAL = self.adversarial_loss(real_preds, valid_d)
            D_LOSS = (LOSS_D_FAKE + LOSS_D_REAL) * 0.5 
            D_LOSS.backward()
            self.optimizer_D.step()

            self.logger.info(f"Batch {i+1}/{len(self.train_dl)} - Discriminator Loss: {D_LOSS.item()}")

            # Train the generator
            self.generator.train()
            self.optimizer_G.zero_grad()
            for p in self.discriminator.parameters():
                p.requires_grad = False

            fake_image = torch.cat([L, generated_abs], dim=1)
            fake_preds = self.discriminator(fake_image)
            LOSS_G_GAN = self.adversarial_loss(fake_preds, valid)
            LOSS_L1 = self.content_loss(generated_abs, abs_) * self.lambda_l1
            LOSS_G = LOSS_G_GAN + LOSS_L1
            LOSS_G.backward()
            self.optimizer_G.step()

            self.logger.info(f"Batch {i+1}/{len(self.train_dl)} - Generator Loss: {LOSS_G.item()}")

            # Accumulate losses
            epoch_d_loss += D_LOSS.item()
            epoch_g_loss += LOSS_G.item()
            num_batches += 1

            # Update progress bar with current loss values
            pbar.set_postfix(D_loss=D_LOSS.item(), G_loss=LOSS_G.item())

        # Average losses for the epoch
        avg_d_loss = epoch_d_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches
        train_loss_discriminator.append(avg_d_loss)
        train_loss_generator.append(avg_g_loss)

        # Scheduler update
        self.scheduler_D.step(D_LOSS)
        d_lr = self.scheduler_D.get_last_lr()
        self.scheduler_G.step(LOSS_G)
        g_lr = self.scheduler_G.get_last_lr()


        self.logger.info(f"Epoch {epoch+1} completed - Avg Discriminator Loss: {avg_d_loss}, Avg Generator Loss: {avg_g_loss}")

        return train_loss_discriminator, train_loss_generator, d_lr, g_lr

    # def visualize_images(self, L, generated_abs, abs_, epoch, batch_idx, show_fig=True, save_images=True):
    #     """
    #     Visualizes and/or saves a batch of generated images and their corresponding ground truth images.
        
    #     Args:
    #         L (torch.Tensor): The L channel of the input images.
    #         generated_abs (torch.Tensor): The generated ab channels from the generator.
    #         abs_ (torch.Tensor): The real ab channels.
    #         epoch (int): The current epoch number.
    #         batch_idx (int): The index of the current batch.
    #         show_fig (bool): Whether to display the images.
    #         save_images (bool): Whether to save the images.
    #     """
    #     L_plot = L[0:4, ...].clone()
    #     generated_abs_plot = generated_abs[0:4, ...].clone()
    #     abs_plot = abs_[0:4, ...]

    #     fake_images = lab_to_rgb(L_plot, generated_abs_plot)
    #     real_images = lab_to_rgb(L_plot, abs_plot)

    #     if show_fig:
    #         print("*" * 100)
    #         print("Generated Images -- Real Images -- Black and White")
    #         plot_batch(fake_images, real_images, L)
    #         self.logger.info("Images visualized and plotted")

    #     if save_images:
    #         # Save images as a batch in the 'training_images' directory
    #         image_filename = os.path.join(self.image_save_dir, f'epoch_{epoch+1}_batch_{batch_idx+1}.png')
    #         self.logger.info(f"Saving images to {image_filename}")
    #         plot_batch(fake_images, real_images, L, show=False, save_path=image_filename)


####################################################################################################################################
# 9

class Validation:
    def __init__(self, generator, discriminator, adversarial_loss, content_loss, lambda_l1, val_dl, device, base_dir, run_dir):

        self.generator = generator
        self.discriminator = discriminator
        self.adversarial_loss = adversarial_loss
        self.content_loss = content_loss
        self.lambda_l1 = lambda_l1
        self.val_dl = val_dl
        self.device = device
        
        # Directory for saving images
        self.image_save_dir = os.path.join(base_dir, run_dir, 'validation_images')
        os.makedirs(self.image_save_dir, exist_ok=True)
    

    def validate_epoch(self, epoch, epochs, show_fig=True, save_images=True):
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

                if show_fig or save_images:
                    visualize_images(L, generated_abs, abs_, epoch, i, self.image_save_dir)
                    show_fig = False
                    save_images = False

                # Discriminator validation
                fake_image = torch.cat([L, generated_abs.detach()], dim=1)
                fake_preds = self.discriminator(fake_image)
                LOSS_D_FAKE = self.adversarial_loss(fake_preds, fake_labels)

                real_images = torch.cat([L, abs_], dim=1)
                real_preds = self.discriminator(real_images)
                LOSS_D_REAL = self.adversarial_loss(real_preds, real_labels)
                D_LOSS = (LOSS_D_FAKE + LOSS_D_REAL) * 0.5 

                # Generator validation
                fake_image = torch.cat([L, generated_abs], dim=1)
                fake_preds = self.discriminator(fake_image)
                LOSS_G_GAN = self.adversarial_loss(fake_preds, real_labels)
                LOSS_L1 = self.content_loss(generated_abs, abs_) * self.lambda_l1
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
# 10

import logging
import os
import matplotlib.pyplot as plt

class Visualization:
    @staticmethod
    def plot_losses(epoch, train_loss_discriminator, val_loss_discriminator, train_loss_generator, val_loss_generator, run_dir, base_dir='training_runs', show_fig=False, save_images=True):
        """
        Plots the training and validation losses for the discriminator and generator, as well as a comparison of both
        generator and discriminator training losses over the epochs. Saves the plots to a subdirectory inside 
        the run directory under the base directory.
        
        Args:
            epoch (int): The current epoch number.
            train_loss_discriminator (list): List of discriminator training losses.
            val_loss_discriminator (list): List of discriminator validation losses.
            train_loss_generator (list): List of generator training losses.
            val_loss_generator (list): List of generator validation losses.
            run_dir (str): The specific directory for the current run (e.g., 'run_1').
            base_dir (str): The base directory where all runs are stored (default: 'training_runs').
            show_fig (bool): If True, will plt.show() plots. Default is False.
            save_image(bool): If True will save plots as pngs to run_dir. Default is True. 
        """
        # Set up a logger specific to this class
        logger = logging.getLogger('Visualization')

        # Create the base directory (e.g., 'training_runs/run_1/loss_plots') if it doesn't exist
        save_dir = os.path.join(base_dir, run_dir, 'loss_plots')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            logger.info(f"Created directory {save_dir} for saving loss plots")

        logger.info("Starting to plot losses")

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

        # Save the plot as an image file in the 'loss_plots' directory inside the run directory
        plot_filename = os.path.join(save_dir, f'loss_plot_epoch_{epoch + 1}.png')
        plt.tight_layout()

        if save_images==True:
            plt.savefig(plot_filename)
            logger.info(f"Saved loss plot for epoch {epoch + 1} at {plot_filename}")
        if show_fig==True:
            # Show the plot (optional, depending on whether you want to display it during training)
            plt.show()
        plt.close()

        logger.info("Completed plotting losses")

####################################################################################################################################
# 11

class GANDriver:
    def __init__(self, generator, discriminator, train_dl, val_dl, optimizer_G, optimizer_D, adversarial_loss, content_loss, 
                 lambda_l1, device, epochs, scheduler_D, scheduler_G, run_dir, base_dir='training_runs'):
        """
        Initializes the GANDriver with models, optimizers, losses, and data loaders for training and validation.
        """
        # Set up a logger specific to this class
        self.logger = logging.getLogger('GANDriver')

        self.device = device
        self.epochs = epochs
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.adversarial_loss = adversarial_loss
        self.content_loss = content_loss
        self.lambda_l1 = lambda_l1

        # Initialize the training and validation classes
        self.trainer = Training(self.generator, self.discriminator, self.optimizer_G, self.optimizer_D,
                                self.adversarial_loss, self.content_loss, self.lambda_l1, train_dl, self.device, 
                                scheduler_D, scheduler_G, run_dir, base_dir)
    
        self.validator = Validation(self.generator, self.discriminator, self.adversarial_loss, 
                                    self.content_loss, self.lambda_l1, val_dl, self.device, base_dir, run_dir)
    
        self.visualizer = Visualization()

        # Lists to store the training and validation losses
        self.train_loss_discriminator = []
        self.train_loss_generator = []
        self.val_loss_discriminator = []
        self.val_loss_generator = []
        self.D_lr = []
        self.G_lr = []
        # Store the run directory for saving images and weights
        self.run_dir = run_dir  
        self.base_dir = base_dir

    def run(self, show_fig=True, save_images=True):
        """
        Runs the training process and saves the model weights after training.

        Args:
            show_fig (bool): Whether to display images during training.
            save_images (bool): Whether to save images during training.

        Reurns:
            dict: A dictionary of training and validation losses for the generator and discriminator
        """
        # Generate a unique run directory
        self.logger.info(f"Starting GAN training run: {self.run_dir}")

        # Iterate over epochs
        for epoch in range(self.epochs):
            print(f"\n========== Epoch {epoch+1}/{self.epochs} ==========")
            self.logger.info(f"Starting epoch {epoch+1}/{self.epochs}")

            # Perform training and validation for the current epoch
            train_d_loss, train_g_loss, d_lr, g_lr = self.trainer.train_epoch(epoch, self.epochs, show_fig=show_fig, 
                                                                  save_images=save_images)
            val_d_loss, val_g_loss = self.validator.validate_epoch(epoch, self.epochs)

            # Append the losses to the corresponding lists
            self.train_loss_discriminator.extend(train_d_loss)
            self.train_loss_generator.extend(train_g_loss)
            self.val_loss_discriminator.extend(val_d_loss)
            self.val_loss_generator.extend(val_g_loss)
            self.D_lr.extend(d_lr)
            self.G_lr.extend(g_lr)

            # Visualize and save the losses
            self.visualizer.plot_losses(epoch, self.train_loss_discriminator, self.val_loss_discriminator,
                                        self.train_loss_generator, self.val_loss_generator, run_dir=self.run_dir, 
                                        base_dir=self.base_dir, show_fig=show_fig, save_images=save_images)

            self.logger.info(f"Completed epoch {epoch+1}/{self.epochs} - Generator Loss: {train_g_loss[-1]}, Discriminator Loss: {train_d_loss[-1]}")

            
            # Training complete, save the model weights
            self.save_model_weights()
            print("\nTraining complete and model weights saved.")
            self.logger.info("Training complete and model weights saved.")

        # Return the losses as a dictionary
        return {
            'train_loss_discriminator': self.train_loss_discriminator,
            'train_loss_generator': self.train_loss_generator,
            'val_loss_discriminator': self.val_loss_discriminator,
            'val_loss_generator': self.val_loss_generator,
            'discriminator_lr': self.D_lr,
            'generator_lr': self.G_lr
        }
    def save_model_weights(self):
        """
        Save the generator and discriminator weights to uniquely named subfolders.

        Args:
            run_dir (str): The specific directory for the current run.
            base_dir (str): Base directory where model weights will be saved.
        """
        # Create the directory to save weights in
        save_dir = os.path.join(self.base_dir, self.run_dir, 'model_weights')

        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)
        self.logger.info(f"Saving model weights to {save_dir}")

        # Define paths for generator and discriminator weights
        save_path_generator = os.path.join(save_dir, 'generator_weights.pth')
        save_path_discriminator = os.path.join(save_dir, 'discriminator_weights.pth')

        # Save the generator weights and optimizer state
        torch.save({'model_state_dict': self.generator.state_dict(),
                    'optimizer_state_dict': self.optimizer_G.state_dict()
                   }, save_path_generator)
    
        # Save the discriminator weights and optimizer state
        torch.save({'model_state_dict': self.discriminator.state_dict(),
                    'optimizer_state_dict': self.optimizer_D.state_dict()
                   }, save_path_discriminator)

        print(f"Generator weights saved to {save_path_generator}")
        print(f"Discriminator weights saved to {save_path_discriminator}")
        self.logger.info(f"Generator weights saved to {save_path_generator}")
        self.logger.info(f"Discriminator weights saved to {save_path_discriminator}")

####################################################################################################################################
# 12

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
# 13

class EntropyLoss(nn.Module):
    def __init__(self, num_bins=256):
        super(EntropyLoss, self).__init__()
        self.num_bins = num_bins

    def forward(self, predicted_ab, ground_truth_ab):
        # Flatten the ab channels
        pred_flat = predicted_ab.view(-1)
        gt_flat = ground_truth_ab.view(-1)

        # Calculate histograms (discretizing the continuous ab channels)
        pred_hist = torch.histc(pred_flat, bins=self.num_bins, min=-1.0, max=1.0)
        gt_hist = torch.histc(gt_flat, bins=self.num_bins, min=-1.0, max=1.0)

        # Normalize the histograms to get probability distributions
        pred_prob = pred_hist / pred_hist.sum()
        gt_prob = gt_hist / gt_hist.sum()

        # Add a small epsilon to avoid log(0)
        epsilon = 1e-8
        pred_prob = pred_prob + epsilon
        gt_prob = gt_prob + epsilon

        # Compute entropy for both predicted and ground truth ab channels
        pred_entropy = -torch.sum(pred_prob * torch.log(pred_prob))
        gt_entropy = -torch.sum(gt_prob * torch.log(gt_prob))

        # Loss is the absolute difference in entropy between predicted and ground truth
        entropy_loss = torch.abs(pred_entropy - gt_entropy)

        return entropy_loss


####################################################################################################################################
# END
####################################################################################################################################
if __name__ == "__Main__":
    pass