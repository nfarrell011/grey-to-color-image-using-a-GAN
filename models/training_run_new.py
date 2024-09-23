#!/usr/bin/python3

import glob
import yaml
import torch
import logging
import os
import pandas as pd
from torch.utils.data import DataLoader
from discriminator import Discriminator
from generator import UNet
from gan_utils_new import *

import warnings

# Suppress specific warning related to CIE-LAB conversion
warnings.filterwarnings("ignore", message=".*negative Z values that have been clipped to zero.*")

# Set up logging
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s %(message)s')

# Function to load configuration from YAML file
def load_config(config_path='params.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Function to select optimizer
def get_optimizer(optimizer_config, model_params):
    opt_type = optimizer_config['type']
    lr = optimizer_config['lr']

    if opt_type == "Adam":
        beta1 = optimizer_config['beta1']
        beta2 = optimizer_config['beta2']
        optimizer = torch.optim.Adam(model_params, lr=lr, betas=(beta1, beta2))

    elif opt_type == "SGD":
        momentum = optimizer_config['momentum']
        optimizer = torch.optim.SGD(model_params, lr=lr, momentum=momentum)

    else:
        raise ValueError(f"Optimizer type '{opt_type}' not recognized. Please choose 'Adam' or 'SGD'.")

    return optimizer

# Main function
def main():
    # Load the configuration from YAML
    config = load_config()

    # Setup device (GPU/CPU)
    if torch.cuda.is_available():
        logging.info("CUDA is available. Using GPU.")
        device = torch.device("cuda")
    else:
        logging.info("CUDA is not available. Using CPU.")
        device = torch.device("cpu")

    # File path from YAML
    coco_path = config['data']['coco_path']
    paths = glob.glob(coco_path + "/*.jpg")  # Grabbing all the image file names

    # Load number of images from config
    num_imgs = config['data']['num_imgs']
    split = config['data']['split']
    train_paths, val_paths = select_images(paths, num_imgs, split)
    logging.info(f"Training set: {len(train_paths)} images")
    logging.info(f"Validation set: {len(val_paths)} images")

    # Image size from YAML
    size = config['data']['image_size']
    train_ds = ColorizationDataset(size, paths=train_paths, split="train")
    val_ds = ColorizationDataset(size, paths=val_paths, split="val")

    # Batch size from YAML
    batch_size = config['training']['batch_size']
    train_dl = DataLoader(train_ds, batch_size=batch_size)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    # Check Tensor Size
    data = next(iter(train_dl))
    Ls, abs_ = data['L'], data['ab']
    assert Ls.shape == torch.Size([batch_size, 1, size, size]) and abs_.shape == torch.Size([batch_size, 2, size, size])

    # Model parameters
    generator = UNet()
    discriminator = Discriminator()

    # Initialize the models
    initializer = ModelInitializer(device, init_type=config['model']['init_type'], gain=config['model']['gain'])
    generator = initializer.init_model(generator)
    discriminator = initializer.init_model(discriminator)

    # Move models to device (GPU/CPU)
    generator.to(device)
    discriminator.to(device)

    # Loss functions from YAML
    adversarial_loss = nn.BCELoss()
    content_loss = PerceptualLoss()  
    l1_loss = nn.L1Loss()
    lambda_l1 = config['training']['lambda_l1']

    # Get optimizer from YAML configuration for both generator and discriminator
    optimizer_G = get_optimizer(config['optimizer'], generator.parameters())
    optimizer_D = get_optimizer(config['optimizer'], discriminator.parameters())

    # Number of epochs from YAML
    epochs = config['training']['epochs']

    # Flags for showing and saving images
    show_fig = config['training']['show_fig']
    save_images = config['training']['save_images']

    # Initialize GANDriver with all parameters from YAML
    driver = GANDriver(
        generator=generator,
        discriminator=discriminator,
        train_dl=train_dl,
        val_dl=val_dl,
        optimizer_G=optimizer_G,
        optimizer_D=optimizer_D,
        adversarial_loss=adversarial_loss,
        content_loss=content_loss,
        lambda_l1=lambda_l1,
        device=device,
        epochs=epochs,
        run_dir=config['output']['run_dir'],
        base_dir=config['output']['base_dir']
    )

    # Run the GAN training and save metrics to CSV after each epoch
    train_results = driver.run(show_fig=show_fig, save_images=save_images)

    # Save training results to CSV
    results_df = pd.DataFrame(train_results)
    result_path = f"{config['output']['base_dir']}/{config['output']['run_dir']}/{config['output']['training_results_csv']}"
    results_df.to_csv(result_path, index=False)
    logging.info(f"Training complete. Results saved to {result_path}.")

if __name__ == "__main__":
    main()