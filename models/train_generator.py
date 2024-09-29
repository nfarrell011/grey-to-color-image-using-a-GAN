import glob
import torch
from torch.utils.data import DataLoader
from discriminator import *
from gan_utils_new import *
import tqdm


# Set the device
if torch.cuda.is_available():
    print("CUDA is available. Running a test on the GPU.")
    device = torch.device("cuda")
else:
    print("CUDA is not available. Running on CPU.")
    device = torch.device("cpu")

# Get the model
generator = Unet()
generator.to(device)

# Set params
size = (256, 256)
batch_size = 16
epochs = 200


# Set the loss
l1_loss = nn.L1Loss()

# Set Optimizer
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
lambda_l1 = 10000
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))

# Set learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=5, verbose=True)

# Get the data
coco_path = "/home/farrell.jo/cGAN_grey_to_color/data/train_sample"
paths = glob.glob(coco_path + "/*.jpg") # Grabbing all the image file names

# Call the function with the desired number of images
num_imgs = 100
train_paths, val_paths = select_images(paths, num_imgs)
print(f"Training set: {len(train_paths)} images")
print(f"Validation set: {len(val_paths)} images")

train_ds = ColorizationDataset(size, paths = train_paths, split = "train")
val_ds = ColorizationDataset(size, paths = val_paths, split = "val")

train_dl = DataLoader(train_ds, batch_size = 16)
val_dl = DataLoader(val_ds, batch_size = 16)

data = next(iter(train_dl))
Ls, abs_ = data['L'], data['ab']
assert Ls.shape == torch.Size([batch_size, 1, 256, 256]) and abs_.shape == torch.Size([batch_size, 2, 256, 256])
print(Ls.shape, abs_.shape)
print(len(train_dl), len(val_dl))

# Use tqdm for the training loop with a progress bar
for epoch in range(epochs):
    epoch_g_loss = 0
    num_batches = 0
    train_loss_generator = []

    pbar = tqdm(train_dl, desc=f"Training Epoch {epoch+1}/{epochs}")
    for i, data in enumerate(pbar):
        L, abs_ = data["L"], data["ab"]
        L, abs_ = L.to(device), abs_.to(device)
        
        # Train the generator
        generator.train()
        optimizer_G.zero_grad()
        generated_abs = generator(L)

        LOSS_L1 = l1_loss(generated_abs, abs_) 
        LOSS_L1.backward()
        optimizer_G.step()

        # Accumulate losses
        epoch_g_loss += LOSS_L1.item()
        num_batches += 1

        # Update progress bar with current loss values
        pbar.set_postfix(G_loss=LOSS_L1.item())

    # Average losses for the epoch
    avg_g_loss = epoch_g_loss / num_batches
    train_loss_generator.append(avg_g_loss)

    scheduler.step(avg_g_loss)

