# Quantum Mottle Denoising Autoencoder

### My Denoising Autoencoder implementation for denoising quantum mottle/noise within chest X-Rays

#### Check out my project presentation for a full demo of how it works: [Project Demo](https://docs.google.com/presentation/d/1ib7q1B89Eghb0NEJpJqkWFSf9YPsaxEV6iwQT6AHKhQ/edit?usp=sharing)!

For this project, I retrieved images from a [dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC) of over 100,000 deidentified chest X-Rays from the NIH Clinical Center.

Citation for source:

Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, Ronald Summers, ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases, IEEE CVPR, pp. 3462-3471, 2017

#### This project consists of three separate, full models for denoising:
 - **DenoisingAutoencoder**: Original Denoising Autoencoder seen in Project Part 3. Uses ***Encoder*** and ***Decoder*** models. Contains three convolutional layers, one fully-connected layer, and a max-pooling reduction factor of 8.

 - **NewDenoisingAutoencoder**: Updated and improved version of the original Denoising Autoencoder model. Uses ***NewEncoder*** and ***NewDecoder*** models. Contains four convolutional layers, one fully-connected layer, and a max-pooling reduction factor of 16.

 - **WindowDenoisingAutoencoder**: Denoising Autoencoder model implementing 'sliding window' approach to denoising. Uses ***WindowEncoder*** and ***WindowDecoder*** models. Like ***NewDenoisingAutoencoder***, it contains four convolutional layers, but it has the additional computational overhead of producing windows of size 64x64.
 
My training, validation, and test sets were split 60%/20%/20%. Each model was trained on 900 pairs of clean/noisy images, validated on 300 pairs of clean/noisy images, and tested on another 300 such pairs. 

The images in each set were chosen at random from the larger dataset I acquired. I initally planned to use more training pairs, but I found minimal benefit to training beyond this number; additionally, training each model took several hours, so adding more pairs was often not worth the time tradeoff. 

Because the images for each set are chosen at random, the test set is generalized enough to extend to any given chest X-Ray from the wider databse. The training and validation sets are largely representative of the overall dataset. Each image pair in every set is distinct from one another, but they are all the same size and consist of (roughly) the same image angle and coverage of the chest.
*As a note, the dataset consists of 1024x1024 images, but to save memory and training time without losing too much image detail, I resized every image to 512x512.*

On the test set for each model, I had the following losses:

**Original model: 0.009, New model: 0.008, Window model: 0.013**

These values may lead one to think that the sliding window implementation had the worst results, but when looking at a visual representation of eah model's denoising, it's clear that ***WindowDenoisingAutoencoder*** was the best model at preserving localized details of all image pairs. 
 
### See how the models visually compare on test data here:
![Images of the clean (original), noisy (with quantum noise), and cleaned X-Rays after running through original model](/OriginalModel.png "Original model output on Test Data")
![Images of the clean (original), noisy (with quantum noise), and cleaned X-Rays after running through new model](/NewModel.png "New model output on Test Data")
![Images of the clean (original), noisy (with quantum noise), and cleaned X-Rays after running through sliding window model](/WindowModel.png "Window model output on Test Data")

Each model is highly effective at removing noise, but they do struggle at fully preserving image details and reconstructing the noisy image to its original detail. The ***NewDenoisingAutoencoder*** model does a better job than the ***DenoisingAutoencoder*** model, which I attribute to its additional convolutional layer. However, it is a bit more computationally intensive, and the training time is not as I found that the sliding window approach is certainly the best way to go about this denoising, as it breaks down the image details into smaller "neighborhoods" rather than evaluating the image shape as a whole. These 512x512 images are clearly too large not to break up into smaller pieces. 

With smaller windows, more convolutional layers, more memory to train with, and perhaps more training and validation pairs, I think a sliding window model can be trained to: 
1. Very effectvely denoise X-Rays with quantum noise (which my model(s) have already shown to be true), and 
2. Preserve image details to the extent that a physician could analyze the reconstructed image with just as much confidence as an original, non-noisy image.

When it came down to it, Google Colaboratory did not have give me the memory capacity to train a windowed model complex enough to preserve image details of 512x512 images. However, I know that by using this approach, such effective denoising is completely attainable.

As for the results of my test set vs. my training and validation sets, this isn't as much of a concern for my project. As stated above, the training, validation, and test sets are all pulled at random from the same larger dataset, so each set is similar to one another (in aggregate, at least). The performance of my models on training data vs. test data is roughly equal (as well as across the validation class). See a visualization of each model's performance on the training data below:

![Images of the clean (original), noisy (with quantum noise), and cleaned X-Rays from training data after running through original model](/TrainOriginalModel.png "Original model output on Training Data")
![Images of the clean (original), noisy (with quantum noise), and cleaned X-Rays from training data after running through new model](/TrainNewModel.png "New model output on Training Data")
![Images of the clean (original), noisy (with quantum noise), and cleaned X-Rays from training data after running through sliding window model](/TrainWindowModel.png "Window model output on Training Data")

---------------------------------------------------------
# Running the Project
I developed and ran my model exclusively within Google Colaboratory. You can either download the Jupyter notebook from here, or just make a copy of my Project on Colabs [Link Here](https://colab.research.google.com/drive/1yNGDgLccqtCsBuoVYbiwM-LXXFNCwLX6?usp=sharing).

### As a note, I only tested my Project within Google Colabs using no hardware accelerator (so using CPU instead of GPU). **Only run this project using CPU**.

To run the model, **run each code block** ***except*** **for the ones marked with**:

## "**Skip the following block if you are running to test my project**"

You will reach a section marked:

## "For individual testing"

Follow the directions listed on this step by putting paths to your clean and noisy image, as well as paths to each model's weights. 

## You can download my weights here:

[Weights for original model](https://drive.google.com/file/d/1cGIv0qslL-fzNQ_y7yj6UtJYGFbmKKTz/view?usp=share_link)

[Weights for new model](https://drive.google.com/file/d/19LiFFDfQQp8xhOBuL_I_QDjyr2Hhwzb3/view?usp=share_link)

[Weights for sliding window model](https://drive.google.com/file/d/1RTQ4mP4T16mqsfu4ddjdiZNA7U_qRA5I/view?usp=share_link)

## And you can download a sample clean and noisy image here:

[Clean Image](https://drive.google.com/file/d/1M3ZCEfoikS6eLSiowyN9t3vV6Gtnn74r/view?usp=share_link)

[Noisy Image](https://drive.google.com/file/d/1RYBhDouNPmAVKpGuBBnEWeE6mVfRdnN-/view?usp=share_link)

### Download all the above links and put them somewhere convenient in your Google Drive (or somewhere locally if you choose to run this locally)

Then, just run the last two blocks to see how the models compare!
