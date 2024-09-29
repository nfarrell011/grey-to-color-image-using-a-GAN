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
from pretrain_generator_class import *

def main():

    # params
    checkpoint_path = "/home/farrell.jo/cGAN_grey_to_color/models/generator_train/Res_full_data_3/gen_weights/checkpoint_epoch_181.pth"
    size = 256
    batch_size = 32
    epochs = 101
    lr = 0.0002 
    beta1 = 0.5
    beta2 = 0.999
    l1_loss = nn.L1Loss()
    run = "Res_full_data_3"
    start_epoch = 0

    # train model
    model = PretrainGenerator(size, batch_size, epochs, lr, beta1, beta2, l1_loss, run, start_epoch)
    model.set_model()
    model.load_state(checkpoint_path)

if __name__ == "__main__":
    main()
    
