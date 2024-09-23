#!/usr/bin/python3


import glob
import torch
from torch.utils.data import DataLoader

from discriminator import Discriminator
from generator import UNet
from gan_utils import *



def main():

    if torch.cuda.is_available():
        print("CUDA is available. Running a test on the GPU.")
        device = torch.device("cuda")
    else:
        print("CUDA is not available. Running on CPU.")
        device = torch.device("cpu")

    # File path
    coco_path = "/Users/mikey/.fastai/data/coco_sample/train_sample"
    paths = glob.glob(coco_path + "/*.jpg") # Grabbing all the image file names

    # Call the function with the desired number of images
    num_imgs = 10
    train_paths, val_paths = select_images(paths, num_imgs)
    print(f"Training set: {len(train_paths)} images")
    print(f"Validation set: {len(val_paths)} images")

    # image size
    size = 256
    train_ds = ColorizationDataset(size, paths = train_paths, split = "train")
    val_ds = ColorizationDataset(size, paths = val_paths, split = "val")

    train_dl = DataLoader(train_ds, batch_size = 4)
    val_dl = DataLoader(val_ds, batch_size = 4)


    ### Setup Dataloader
    train_dl = DataLoader(train_ds, batch_size = 4)
    val_dl = DataLoader(val_ds, batch_size = 4)

    # Check Tensor Size
    data = next(iter(train_dl))
    Ls, abs_ = data['L'], data['ab']
    #assert Ls.shape == torch.Size([4, 1, 256, 256]) and abs_.shape == torch.Size([4, 2, 256, 256])


    ### Params
    # Assuming UNet is already defined as per the code above
    generator = UNet()
    discriminator = Discriminator()

    # Initialize the models
    initializer = ModelInitializer(device, init_type='norm', gain=0.2)
    generator = initializer.init_model(generator)
    discriminator = initializer.init_model(discriminator)

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    # Loss functions
    adversarial_loss = nn.BCELoss()  
    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()
    perceptual_loss = PerceptualLoss

    # Optimizers
    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999
    lambda_l1 = 100

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    # training params
    epochs = 2
    batch_size = 16

    ### Training
    driver = GANDriver(
        generator=generator,
        discriminator=discriminator,
        train_dl=train_dl,
        val_dl=val_dl,
        optimizer_G=optimizer_G,
        optimizer_D=optimizer_D,
        adversarial_loss=adversarial_loss,
        l1_loss=l1_loss,
        lambda_l1=lambda_l1,
        device=device,
        epochs=epochs
    )

    # Run the GAN training
    driver.run()

if __name__ =="__mina__":
    main()