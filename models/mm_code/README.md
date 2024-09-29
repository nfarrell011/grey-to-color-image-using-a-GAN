    Nelson Farrell & Michael Massone  
    Converting Grey Scale to Color using A GAN  
    09.27.2024
    CS 7180 Advanced Perception - Prof. Bruce Maxwell, PhD.

---

# Grayscale to Color Image Conversion using GAN

## Project Overview
This project implements a Generative Adversarial Network (GAN) to convert grayscale images into color images. The GAN consists of a U-Net-based generator and a patch-based discriminator. The generator receives grayscale images and produces colorized images in the LAB color space. The discriminator helps improve the generator's performance by distinguishing between real and generated images.

The model is trained on a dataset of natural images, and the results include both qualitative and quantitative evaluations of the colorization performance.

## Directory Structure
```
├── data/                      # Directory for sample images
│
├── src/                         # Main source code directory
│   ├── gan_utils_new.py         # GAN utilities for training and evaluation
│   ├── training_run.py          # Script to train the GAN model
│   ├── models/
│   │   ├── generator.py         # Generator (U-Net) model definition
│   │   └── discriminator.py     # PatchGAN Discriminator model definition
│   └── params.yaml              # Configuration file for the training setup
│
├── eval/                        # Main source code directory
│   └── validation.ipynb.py      # Notebook for visually comparing model inference to groundtruth
│
├── training_runs/               # Directory for storing model results
│   ├── run_1/                   # Directory for the first run
│   │   ├── model_weights/       # Saved model weights and optimize state for generator and discriminator
│   │   ├── training_images/     # Sample images generated during training
|   |   ├── validation_images/   # Sample images generated during validation
│   │   ├── config.yml           # Configuration file used for this run
│   │   ├── training_results.csv # CSV file storing the training and validation metrics
│   │   └── run_1.log            # Log file for this training run
│   └── run_2/                   # Directory for the second run (similar structure to run_1)
│
├── gan_env.yml                  # Conda environemnt yaml.
└── README.md                    # This README file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/grayscale-to-color-gan.git
   cd grayscale-to-color-gan
   ```

2. Set up the conda environment:
   ```bash
   conda env create --file gan_env.yml
   conda activate GAN_env_CUDA11_8  # Activate the environment
   ```

## Load Dataset

1. Prepare your dataset in the `image_enhancement/data/` folder.

## Generator Pre-Training

Optional pretraining of Generator model. 

1. 

## GAN Training

To train the GAN model on your dataset:

1. Set the `data_path` in `config.yml` to the directory containing your dataset.
   
2. Update the configuration in `params.yml` to match your dataset and training requirements.

3. Run the training script:
   ```bash
   python image_enhancement/training_run_new.py
   ```

## Results

After training, the results for each run are stored in the `training_runs/` directory. For example, after running the model, `run_1/` will contain:

- `model_weights/`: Directory containing the generator and discriminator weights and optimizer states saved during training.
- `training_images/`: Directory containing sample images generated during training.
- `validation_images/`: Directory containing sample images generated during training.
- `config.yml`: Configuration file used for the run.
- `training_results.csv`: CSV file storing the training and validation metrics (e.g., losses).
- `run_1.log`: Log file for the training process.

## Inference with Trained Model

To use the trained model for inference:

1. Load the saved weights from `training_runs/run_x/model_weights/` into the model.
2. Run the inference script with new grayscale images.
3. Save the output colorized images.

## Acknowledgements

1. Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network {}
2. Image Colorization using Generative Adversarial Networks {https://arxiv.org/pdf/1803.05400}
3. Image-to-Image Translation with Conditional Adversarial Networks {https://arxiv.org/pdf/1611.07004}
4. Color Black and White Image with U-Net and conditional GAN - A Tutorial https://towardsdatascience.com/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8

## License

This project is licensed under the MIT License.

---
