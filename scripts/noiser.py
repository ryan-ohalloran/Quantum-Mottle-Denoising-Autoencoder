#!/usr/bin/env python3

import cv2
import numpy as np

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

add_poisson_noise(1000., "image.png", "yy.png")