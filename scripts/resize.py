#!/usr/bin/env python3

import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

TRAINING_PATH   = 'training/'
VALIDATION_PATH = 'validation/'
TEST_PATH       = 'test/'

def resize_images(directory):
    for i, filename in enumerate(os.listdir(directory)):
        if filename.endswith('.png'):
            print(f'{i}: {filename}')
            # Open the image file
            with Image.open(os.path.join(directory, filename)) as img:
                # Resize the image to 512x512 pixels
                img_resized = img.resize((512, 512))
                # Save the resized image with the new size
                img_resized.save(os.path.join(directory, filename))

# Usage
resize_images(TEST_PATH)
