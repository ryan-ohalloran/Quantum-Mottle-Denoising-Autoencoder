# Quantum Mottle Denoising Autoencoder

My denoising autoencoder architecture consists of two primary components: the encoder and the decoder. I chose to give my encoder three convolutional layers and a fully connected layer. This allows it to learn a compressed representation of input image (the noisy image). The decoder consists of four transposed convolutional layers. These allow it to reconstruct a cleaned version of the noisy input image. I decidd to go with mean squared error (MSE) for my loss function. MSE is pretty standard for images like this, and it's commonly used in similar projects. It also is good at penalizing large differences between predicted and true images to help minimize noise in the image. For my optimization algorithm, I chose the Adam optimizer. Aside from being an optimizer we've used in class, the way the Adam optimizer adapts the learning rate based on the gradient of its error with respect to the weights is suitable for this project.

This all seemed good to me, but once I checked out the cleaned image vs. the noisy and original (clean) images, I realized it wasn't as good as I thought. 
See that image here:
![Images of the clean (original), noisy (with quantum noise), and cleaned (after runnning through model) X-rays](/output.png "Output on Test Data")

I think my main issue is just fine-tuning my encoder and decoder to produce better output. The dataset I drew from for these images gave them in 1024x1024 format, but that was just too much data to process. Because of this, I decided to resize my images down to 512x512. This definitely took away some accuracy, but not to the extent seen by the above images. With more trial and error, I think I can make these a lot better. The loss values I was getting led me to believe my model was working really, really well, but when I looked at the output images at the end, I realized how much precision I had lost. Perhaps more training/more variation in images would be best. 

I did do a little testing to find a saturation point at which adding additional images no longer made sense, and although I definitely didn't find an exact point, I found that cutting my original break-down of 7,200 pairs in training, 2,400 pairs in validation, and 2,400 pairs in test into 1/4 (yielding 1,800 training pairs, 600 validation pairs, and 600 test pairs) was adequate. I think that my main issue now is just fine-tuning my algorithm, greatly improving its ability to reconstruct the images, and reducing accuracy loss. This is a tough task, but it's also something I know I am capable of achieving. 

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
