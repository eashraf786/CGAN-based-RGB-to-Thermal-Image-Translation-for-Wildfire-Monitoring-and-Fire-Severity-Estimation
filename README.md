# CGAN-based-RGB-to-Thermal-Image-Translation-for-Wildfire-Monitoring-and-Fire-Severity-Estimation

This project implements a Generative Adversarial Network (GAN) for translating RGB images to infrared (thermal) images. The model is trained on paired RGB and thermal images from the FLAME2 dataset, with a focus on fire and smoke detection scenarios.

## Project Structure

```
RGB2IR-Translation/
├── models/
│   ├── generator.py            # Generator model architecture
│   ├── discriminator.py        # Discriminator model architecture
│   ├── gan.py                  # GAN training logic
    └── losses.py               # Custom loss functions
├── data/
│   ├── dataset.py              # Dataset loading and preprocessing
│   ├── labels.py               # Label processing utilities
    └── Frame Pair Labels.txt   # Labels of frame pairs from FLAME2 Dataset
├── notebooks/
│   ├── test-cgan-model.ipynb   # Model Evaluation and sample translations
│   └── fire_severity_3.ipynb   # Unsupervised Segmentation using HDBSCAN clustering and Fire Severity Estimation
├── requirements.txt
└── README.md
```

## Features

- Conditional GAN architecture for RGB to IR image translation
- Custom loss functions combining adversarial and L1 losses
- Support for fire and smoke detection scenarios
- Comprehensive evaluation metrics
- Jupyter notebooks for model evaluation and fire severity estimation

## Requirements

The project requires the following dependencies:
- TensorFlow >= 2.8.0
- NumPy >= 1.20.0
- Matplotlib >= 3.5.0
- scikit-image >= 0.19.0
- scikit-learn >= 1.0.0
- OpenCV >= 4.5.0
- Jupyter and related packages

Install the dependencies using:
```bash
pip install -r requirements.txt
```

## Data

The project uses the FLAME2 dataset, which contains:
- RGB images (254p resolution)
- Thermal images (254p resolution)
- Frame pair labels indicating fire and smoke presence

## Model Architecture

The model consists of:
1. Generator:
   - U-Net style architecture with skip connections
   - Multiple downsampling and upsampling layers
   - Batch normalization and dropout for regularization

2. Discriminator:
   - PatchGAN architecture
   - Multiple convolutional layers
   - Leaky ReLU activation

## Training

The model is trained using:
- Adversarial loss for realistic image generation
- L1 loss for pixel-wise accuracy
- Adam optimizer with beta1=0.5
- Learning rate of 2e-4

## Evaluation

The model is evaluated using:
- Visual comparison of generated images
- Fire and smoke detection accuracy
- Structural similarity metrics
- Peak signal-to-noise ratio
