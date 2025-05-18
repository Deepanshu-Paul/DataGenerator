
# Data Generator

A tool that generates synthetic tabular data using a sample. This code uses Variational Auto Encoder.
This code is a Flask web application that allows users to generate synthetic tabular data using a trained Variational Autoencoder (VAE) model.

Key Features:
Provides a simple web form where users enter the number of synthetic rows to generate.

Loads a pre-trained VAE model and preprocessor (joblib).

Samples latent variables from a normal distribution and decodes them into synthetic data.

Inversely transforms the data to restore original features (both numerical and categorical).

Outputs the generated data as a downloadable CSV file.

What is a Variational Autoencoder (VAE)?
A Variational Autoencoder (VAE) is a type of deep generative model used to learn and generate new data similar to a training dataset.

It has three main parts:
Encoder: Compresses input data into a smaller "latent" representation (a vector).

Reparameterization Trick: Introduces randomness to sample points from a distribution.

Decoder: Reconstructs data from the sampled latent vector.

Why Use VAE?
VAEs learn a smooth, continuous latent space, making it easy to sample new, realistic data.

Useful for generating new examples, anomaly detection, and data augmentation.

## Flowchart


![ChatGPT Image May 17, 2025, 08_12_56 PM](https://github.com/user-attachments/assets/dfed83e3-8e65-4019-a6e9-908423724f49)
