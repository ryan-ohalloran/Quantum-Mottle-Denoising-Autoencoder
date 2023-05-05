# Quantum Mottle Denoising Autoencoder

My denoising autoencoder architecture consists of two primary components: the encoder and the decoder. I chose to give my encoder three convolutional layers and a fully connected layer. This allows it to learn a compressed representation of input image (the noisy image). The decoder consists of four transposed convolutional layers. These allow it to reconstruct a cleaned version of the noisy input image. I decidd to go with mean squared error (MSE) for my loss function. MSE is pretty standard for images like this, and it's commonly used in similar projects. It also is good at penalizing large differences between predicted and true images to help minimize noise in the image. For my optimization algorithm, I chose the Adam optimizer. Aside from being an optimizer we've used in class, the way the Adam optimizer adapts the learning rate based on the gradient of its error with respect to the weights is suitable for this project.

My project doesn't have classify images 'correctly' or 'incorrectly', but rather reconstructs an input image. So, one way to measure how well my architecture is learning is through watching the trend of the loss value as the network trains. After much trial and error, I got the loss generated from my training to decrease with more and more training. Here is the output of my last epoch of training:  
"""   
100it [02:32,  1.44s/it][5, 100] loss: 0.006288994981441647  
200it [05:04,  1.47s/it][5, 200] loss: 0.005869287438690663  
300it [07:36,  1.51s/it][5, 300] loss: 0.005667370993178338  
400it [10:17,  1.46s/it][5, 400] loss: 0.00651495382655412  
500it [12:50,  1.45s/it][5, 500] loss: 0.005999319422990083  
600it [15:26,  1.46s/it][5, 600] loss: 0.0059178611822426315  
700it [18:00,  1.45s/it][5, 700] loss: 0.005194941683439538   
800it [20:33,  1.51s/it][5, 800] loss: 0.006572313182987273   
900it [23:07,  1.52s/it][5, 900] loss: 0.004619200051529333  
1000it [25:39,  1.55s/it][5, 1000] loss: 0.006383404566440732  
1100it [28:13,  1.63s/it][5, 1100] loss: 0.004809260834008455  
1200it [30:43,  1.50s/it][5, 1200] loss: 0.008199772147927433  
1300it [33:14,  1.43s/it][5, 1300] loss: 0.006560161749366671  
1400it [35:45,  1.52s/it][5, 1400] loss: 0.006359847865533084  
1500it [38:24,  1.47s/it][5, 1500] loss: 0.007291274755261839  
1600it [41:00,  1.45s/it][5, 1600] loss: 0.006896869618212804  
1700it [43:31,  1.50s/it][5, 1700] loss: 0.004838906810618937  
1800it [46:05,  1.54s/it][5, 1800] loss: 0.005278926562750712  
  
Validation loss: 0.006  
Saving model state...  
"""  
Compare this with the first epoch:  
"""  
100it [03:22,  2.04s/it][1, 100] loss: 0.04866583268158138  
200it [06:34,  1.87s/it][1, 200] loss: 0.033711762027814986  
300it [09:49,  2.00s/it][1, 300] loss: 0.026005547689273954  
400it [13:11,  2.24s/it][1, 400] loss: 0.021628693398088216  
500it [16:25,  1.92s/it][1, 500] loss: 0.018903119075112045  
600it [19:42,  2.19s/it][1, 600] loss: 0.0175606380822137  
700it [23:35,  2.40s/it][1, 700] loss: 0.016079862266778946  
800it [27:29,  2.11s/it][1, 800] loss: 0.016644945931620896  
900it [31:13,  2.23s/it][1, 900] loss: 0.01443642100552097  
1000it [35:01,  2.20s/it][1, 1000] loss: 0.014933602837845682  
1100it [38:47,  2.27s/it][1, 1100] loss: 0.01508091608993709  
1200it [42:33,  2.21s/it][1, 1200] loss: 0.014850337542593479  
1300it [46:27,  2.19s/it][1, 1300] loss: 0.013409307724796236  
1400it [50:14,  2.29s/it][1, 1400] loss: 0.016770283421501518  
1500it [54:11,  2.26s/it][1, 1500] loss: 0.014509808304719627  
1600it [58:01,  2.29s/it][1, 1600] loss: 0.011522051955107599  
1700it [1:01:50,  2.25s/it][1, 1700] loss: 0.01158424916677177  
1800it [1:05:40,  2.19s/it][1, 1800] loss: 0.010968255084007979  
"""  
And you can clearly see the progression of my model's training. The best validation loss was 0.006. 
Running on the test data led to a test loss of 0.006. 
This all seemed good to me, but once I checked out the cleaned image vs. the noisy and original (clean) images, I realized it wasn't as good as I thought. 
See that image here:
![Images of the clean (original), noisy (with quantum noise), and cleaned (after runnning through model) X-rays](/output.png "Output on Test Data")

I think my main issue is just fine-tuning my encoder and decoder to produce better output. The dataset I drew from for these images gave them in 1024x1024 format, but that was just too much data to process. Because of this, I decided to resize my images down to 512x512. This definitely took away some accuracy, but not to the extent seen by the above images. With more trial and error, I think I can make these a lot better. The loss values I was getting led me to believe my model was working really, really well, but when I looked at the output images at the end, I realized how much precision I had lost. Perhaps more training/more variation in images would be best. 

I did do a little testing to find a saturation point at which adding additional images no longer made sense, and although I definitely didn't find an exact point, I found that cutting my original break-down of 7,200 pairs in training, 2,400 pairs in validation, and 2,400 pairs in test into 1/4 (yielding 1,800 training pairs, 600 validation pairs, and 600 test pairs) was adequate. I think that my main issue now is just fine-tuning my algorithm, greatly improving its ability to reconstruct the images, and reducing accuracy loss. This is a tough task, but it's also something I know I am capable of achieving. 

---------------------------------------------------------
# Running the Project
I developed and ran my model exclusively within Google Colaboratory. You can either download the Jupyter notebook from here, or just make a copy of my Project on Colabs [Link Here](https://colab.research.google.com/drive/1yNGDgLccqtCsBuoVYbiwM-LXXFNCwLX6?usp=sharing).

### As a note, I only tested my Project within Google Colabs using no hardware accelerator (so using CPU instead of GPU). **Only run this project using CPU**.

To run the model, run each code block except for the ones marked with:

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
