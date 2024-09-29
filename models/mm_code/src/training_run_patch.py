#!/usr/bin/python3

import glob
import yaml
import torch
import logging
import os
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from discriminator import PatchDiscriminator
from generator import UNet
from gan_utils_patch import *

import warnings

# Suppress specific warning related to CIE-LAB conversion
warnings.filterwarnings("ignore", message=".*negative Z values that have been clipped to zero.*")

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
        weight_decay = optimizer_config['weight_decay']
        optimizer = torch.optim.Adam(model_params, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)

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
    
    # Define the directory path based on the config (run_dir inside base_dir)
    output_dir = os.path.join(config['output']['base_dir'], config['output']['run_dir'])
    print(f"Output_dir: {output_dir}")
    
    # Create the directories if they don't already exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging: Save the log file inside the run_dir
    log_filename = f"{config['output']['run_dir']}.log"
    logging_filepath = os.path.join(output_dir, log_filename)
    logging.basicConfig(filename=logging_filepath, level=logging.INFO, format='%(asctime)s %(message)s')
    
    # Save the configuration dictionary to a YAML file inside the run_dir
    yaml_filepath = os.path.join(output_dir, 'config.yml')
    with open(yaml_filepath, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    
    print(f"Configuration saved to {yaml_filepath}")
    print(f"Logging to {logging_filepath}")

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
    discriminator = PatchDiscriminator()

    # Initialize the models
    init_D = ModelInitializer(device, init_type=config['init_D']['init_type'], gain=config['init_D']['gain'])
    discriminator = init_D.init_model(discriminator)
    init_G = ModelInitializer(device, init_type=config['init_G']['init_type'], gain=config['init_G']['gain'])
    generator = init_G.init_model(generator)
    

    # Move models to device (GPU/CPU)
    generator.to(device)
    discriminator.to(device)

    # Loss functions from YAML
    adversarial_loss = nn.BCEWithLogitsLoss().to(device)
    content_loss = nn.L1Loss().to(device)
    lambda_l1 = config['training']['lambda_l1']

    # Get optimizer from YAML configuration for both generator and discriminator
    optimizer_G = get_optimizer(config['optimizer_G'], generator.parameters())
    optimizer_D = get_optimizer(config['optimizer_D'], discriminator.parameters())

    # Learning rate scheduler
    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, 
                                                       mode=config['scheduler_G']['mode'], 
                                                       factor=config['scheduler_G']['factor'], 
                                                       patience=config['scheduler_G']['patience'],
                                                       verbose=config['scheduler_G']['verbose'])
    scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode=config['scheduler_D']['mode'], 
                                                       factor=config['scheduler_D']['factor'], 
                                                       patience=config['scheduler_D']['patience'],
                                                       verbose=config['scheduler_D']['verbose'])

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
        scheduler_D=scheduler_D,
        scheduler_G=scheduler_G,
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