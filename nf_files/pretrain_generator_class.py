"""
Nelson Farrell & Michael Massone
Image Enhancement: Colorization - cGAN
CS 7180 Advanced Perception
Bruce Maxwell, PhD.
09-28-2024

This file contains a class that will create and train U-Net for the task of colorization.
The U-Net will created will have pretrained ResNet18 backbone.
"""
##################################################### Packages ###################################################################
import os
import glob
import time
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb

import glob
import torch
from torch.utils.data import DataLoader
#from discriminator import *
from gan_utils_new import *
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet

##################################################### Helper Function ################################################################
def build_res_unet(n_input=1, n_output=2, size=256):
    """
    Builds ResNet18 based U-Net
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = create_body(resnet18, pretrained=True, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size)).to(device)
    return net_G

####################################################### Class def ####################################################################
class PretrainGenerator:
    """
    Class to pretrain the generator network
    """
    def __init__(self, image_size = 256, batch_size = 32, epochs = 100, lr = 0.0002, beta1 = 0.5, beta2 = 0.999, loss = nn.L1Loss(), run = "training_run", start_epoch = 0):
        """
        Initializes PreTrainGenerator classes with all default values.
        See methods to perform sets.
        """
        self.image_size = image_size
        self.batch_size = 256
        self.lr = lr
        self.beta = beta1
        self.beta2 = beta2
        self.loss = loss
        self.run = run
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.optimizer = None
        self.scheduler = None

        self.start_epoch = start_epoch
        self.epochs = epochs
        self.avg_loss = 0

        self.train_ds = None
        self.val_ds = None
        self.train_dl = None
        self.val_dl = None

        self.train_loss_generator = []
        self.val_loss_generator = []
        
    def set_model(self, model:callable = None, use_res_net:bool = True) -> None:
        """
        Set the generator model and optimizer, default is to use a U-Net with a ResNet18 backbone
        """
        if use_res_net:
            body = create_body(resnet18, pretrained=True, n_in=1, cut=-2)
            net_G = DynamicUnet(body, 2, (self.size, self.size)).to(device)
            self.model = net_G
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
            return
        else:
            self.model = model
            self.model.to(self.device)
            return

    def load_state(self, path_to_checkpoint:str) -> None:
        """
        Loads a previous model state
        """
        try:
            checkpoint = torch.load(path_to_checkpoint)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch']
            self.avg_loss = checkpoint['loss']
            print(f"Model state loaded successfully!")
        except FileNotFoundError as e:
            print("Error loading generator weights!")
        return

    def set_data_loaders(self, path_to_data:str, perform_checks:bool = True) -> None:
        """
        Set up the dataloaders
        """
        self.train_ds = ColorizationDataset(size, paths = train_paths, split = "train")
        self.val_ds = ColorizationDataset(size, paths = val_paths, split = "val")
        self.train_dl = DataLoader(train_ds, batch_size = self.batch_size)
        self.val_dl = DataLoader(val_ds, batch_size = self.batch_size)

        if perform_checks:
            data = next(iter(self.train_dl))
            Ls, abs_ = data['L'], data['ab']
            assert Ls.shape == torch.Size([self.batch_size, 1, self.size, self.size]) and abs_.shape == torch.Size([self.batch_size, 2, self.size, self.size])
            print(Ls.shape, abs_.shape)
            print(len(train_dl), len(val_dl))

        return

    def train_loop(self, epoch) -> None:
        """
        Performs the train loop tracking train loss
        """
        epoch_train_loss = 0
        num_batches = 0

        # Train Loop
        pbar = tqdm(self.train_dl, desc=f"Training Epoch {self.start_epoch}/{self.start_epoch + self.epochs}")
        for i, data in enumerate(pbar):
            L, abs_ = data["L"], data["ab"]
            L, abs_ = L.to(self.device), abs_.to(self.device)
    
            # Train the generator
            self.model.train()
            self.optimizer.zero_grad()
            generated_abs = self.model(L)
    
            LOSS = self.loss(generated_abs, abs_) 
            LOSS.backward()
            self.optimizer.step()
    
            # Accumulate losses
            epoch_train_loss += LOSS.item()
            num_batches += 1
    
            # Update progress bar with current loss values
            pbar.set_postfix(G_loss=LOSS.item())
    
        # Average losses for the epoch
        avg_train_loss = epoch_train_loss / num_batches
        self.train_loss_generator.append(avg_train_loss)
        print(f"The average loss for epoch: {epoch} - {avg_train_loss}")
    
        self.scheduler.step(avg_train_loss)

    def val_loop(self, epoch) -> None:
        """
        Performs the val loop tracking val loss
        """
        with torch.no_grad():
            num_batches = 0
            epoch_val_loss = 0
            self.model.eval()
            
            pbar = tqdm(self.val_dl, desc=f"Validation Epoch {self.start_epoch}/{self.start_epoch + self.epochs}")
            for i, data in enumerate(pbar):
                L, abs_ = data["L"], data["ab"]
                L, abs_ = L.to(self.device), abs_.to(self.device)
    
                 # Evaluate the generator
                generated_abs = generator(L)
                LOSS = self.loss(generated_abs, abs_) 
        
                # Accumulate losses
                epoch_val_loss += LOSS.item()
                num_batches += 1
        
                # Update progress bar with current loss values
                pbar.set_postfix(G_loss=LOSS.item())
    
                # Create the directory to save iamges in
                image_save_dir = f"/home/farrell.jo/cGAN_grey_to_color/models/generator_train/{self.run}/val_images/"
                os.makedirs(image_save_dir, exist_ok=True)
                image_save_path = image_save_dir + f"epoch_{epoch}.png"
                
                if epoch % 10 == 1:
                    self.plot_batch(L, generated_abs, abs_, False, image_save_path)
        
        # Calculate average validation loss
        avg_val_loss = epoch_val_loss / num_batches
        self.val_loss_generator.append(avg_val_loss)
        print(f"Avg Validation Loss: {avg_val_loss}")
        return

    def plot_batch(L, generated_abs, real_abs, show=True, save_path=None) -> None:
        """
        Plot or save 4 images.
    
        Args:
            fake_imgs (list): List of generated images.
            real_imgs (list): List of real images.
            L (list): List of black and white images (L channel).
            show (bool): If True, display the images. Defaults to True.
            save_path (str): Path to save the images. If None, the images are not saved. Defaults to None.
        """
        L_plot = L[0:4, ...].clone()
        generated_abs_plot = generated_abs[0:4, ...].clone()
        abs_plot = real_abs[0:4, ...]
    
        fake_imgs = lab_to_rgb(L_plot, generated_abs_plot)
        real_imgs = lab_to_rgb(L_plot, abs_plot)
    
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
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()  # Close the plot to free up memory

    def plot_losses(self, epoch) -> None:
        """
        Generates and saves loss versus epoch plot
        """
        # Create fig
        figs_save_dir = f"/home/farrell.jo/cGAN_grey_to_color/models/generator_train/{self.run}/loss_figs/"
        os.makedirs(figs_save_dir, exist_ok=True)
        figs_save_path = figs_save_dir + f"epoch_{epoch}.png"
                
        # Ensure the directory exists
        epoch_range = range(self.start_epoch, epoch + 1)
        plt.plot(epoch_range, self.train_loss_generator, c = "b", label = "Train Loss")
        plt.plot(epoch_range, self.val_loss_generator, c = "r", label = "Val Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figs_save_path)
        plt.close

    def save_model_state(self, epoch:int) -> None:
        """
        Saves the current model state
        """
        # Create the directory to save weights in
        state_save_dir = f"/home/farrell.jo/cGAN_grey_to_color/models/generator_train/{self.run}/gen_weights/"
    
        # Ensure the directory exists
        os.makedirs(state_save_dir, exist_ok=True)
    
        # Define paths for generator and discriminator weights
        state_save_path = os.path.join(state_save_dir, f'checkpoint_epoch_{epoch}.pth')
    
        # Save the model weights
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.train_loss_generator[-1],
        }, state_save_path)
        print(f"Model state saved to: {state_save_path}")
        
    def train_model(self) -> None:
        """
        Trains the model
        """
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            self.train_loop(epoch)
            self.val_loop(epoch)
            if epoch % 10 == 1:
                self.plot_losses()
                self.save_model_state()
        
    def set_optiminer(self) -> None:
        """
        Method to set up the optimizer
        """
        raise NotImplementedError

    def set_scheduler(self) -> None:
        """
        Method to set up the scheduler
        """
        raise NotImplementedError

if __name__ == "__main__":
    pass

















        
            