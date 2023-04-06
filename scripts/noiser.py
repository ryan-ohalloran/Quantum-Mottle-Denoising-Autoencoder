#!/usr/bin/env python3

import cv2
import numpy as np
import os

PATH = '.'
NOISE_FACTOR = 900.0
# BATCH_PATH = '/Users/ryanohalloran/Documents/JuniorS2/NeuralNetworks/ProjectGithub/Quantum-Mottle-Denoising-Autoencoder/test'

def add_poisson_noise(noiseFactor: float, inputFileName: str, outputFileName: str)->None:
    '''
    Creates noisy image with file name outputFileName from file name inputFileName given noiseFactor
    (higher noiseFactor gives noisier image)
    '''
    # Load the image
    img = cv2.imread(inputFileName, cv2.IMREAD_GRAYSCALE)

    # Add Poisson noise to the image
    noisy_img = np.random.poisson(img / noiseFactor * 250) / 250.0 * noiseFactor

    # Convert the image back to uint8 format
    noisy_img = noisy_img.astype(np.uint8)

    # Save the noisy image
    cv2.imwrite(outputFileName, noisy_img)

# Execute noiser
for file in os.listdir(PATH):
    if file.endswith('.png') and not file.startswith('noisy'):
        add_poisson_noise(NOISE_FACTOR, file, f'noisy_{file}')
