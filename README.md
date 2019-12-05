# Adversarial-Autoencoder-Experiments

## Summary

This repository has a series of experiments using some of the adversarial autoencoders (AAEs) described in [Adversarial Autoencoders](https://arxiv.org/pdf/1511.05644.pdf); I also found parts of the [Wasserstein Auto-encoders](https://arxiv.org/pdf/1711.01558.pdf) very useful. The experiments include

* the MNSIT dataset
  * 2D Gaussian
  * mixed (umbrella) 2D Gaussian
  * mixed (umbrella) 2D Gaussian
* the LFW dataset
  * a 400 dimensional Gaussian
* the CIFAR10 dataset
  * transfer classification with LFW encoding

## Implementation

All of the networks are implemented using PyTorch; while I used the papers for guidance, I tested a range of encoder/decoder/discriminator architectures before settling on the ones included here (I don't include all of the various tests). For better or worse, most of the content resides in a few Jupyter Notebooks (including some long term training), and a collection of utilitiy files are used to try to improve the readability.

SciKit-Learn and NumPy/SciPy are also used as needed, and all visualizations are made using matplotlib and imageio.

## Datasets

The MNIST and CIFAR10 datasets were obtained on the fly as torchvision datasets. The LFW data was downloaded from [Labeled Faces in the Wild Home](http://vis-www.cs.umass.edu/lfw/), and I paired down the LFW data to include a single image per person to reduce training time for testing model architectures. This does appear to have a detrimental impact on the quality of the latent space, and I'm planning on training a version of the model on a larger dataset (I might upload this once I've completed the training).

## Overview of Experiments

#### MNIST Dataset

Here's some of my thoughts/notes on the experiments that I ran on MNIST. I tried three different prior distributions; here's a comparison of the three distributions colored by the label information.

![image here]()

I also saved out the learned manifold for the 2D Gaussian and mixed (umbrella) Gaussian.

![image here]()

I did note that the reconstructions struggled a bit with a 2-dimensional latent space (see below). I know from previous tests (not included here) that altering the encoder/decoder only by increasing the dimension of the latent space drastically improves the reconstruction (which makes sense), but unforunately, it isn't as easy to visualize.

![image here]()


It was interseting for me to note how the autoencoder tries to group not just numbers with the similar shapes together but also information beyond what is. By including the label information (in a semi-supervised scheme), we can guide how the autoencoder groups the data (for example along each mode of the mixed distribution) according to the information it finds relavent to the label. This allows the variation within the model to capture the information that the model does not associate with the label (the so-called style information) such as orientation, handwriting style, etc. Below is replication of one of the images in the paper in which we sample the mixed 2D (umbrella) Gaussian distribution along each mode (which corresponds to each row).

![image here]()

I did notice that the learned manifold (not shown here) tended to be a have a poorer quality (visually) when  using the labeled information as numbers which the autoencoder may not naturally group together are forced to be nearby. This makes me wonder if some care is needed when using the network architecture which feeds the labeled information to the discrimintor (unless you have some apriori information which can help guide you in deciding the prior distribution).

The observations in both of the previous paragraphs are interesting to me and I'm still digesting it, but I think they give good motivation to the (semi) supervised network also propsed. Using a categorical distribution to guide the label information seems like a natural way to avoid imposing too much assumptions on the latent space while still allowing the model to separate the information in the data relevent to the label and to the "style."

I'm excited to implement the (semi) supervised network setup next to explore how this can be used to guide the generative capabilities of auto-encoder, especially given my observations from my next experiment.


#### LFW Datasets

For these experiment, I spent alot of time hand tuning the encoder/decoder in order to get a high fidelity reconstruction. While there are plenty of examples out there of such networks, copy/pasting them doesn't give a lot of intuition or insight as to how they are choosen. Once I had a decently working autoencoder, I incorporated the adversarial component. It took some experimentation to find the right balance of making the discriminator a strong enough learner to challenge the encoder while not overpowering the autoencoder portion of the network and preventing the encoder from learning the prior distribution. Below you can see examples of the reconstruction that the AAE gives on images it was traineed on.

![image here]()

The latent space I used for the facial data is 400-dimensional, so I tested some different methods for understanding/exploring the latent space. The simpliest was to try applying the autoencoder to out of sample facial images and to try randomly sampling from the latent distribution. Below I show both of these images. As you can see, while strong facial features are present in both examples, a great deal of important imformation is not present. I believe this is a result of the small dataset that I trained on (about 5750 images), and I'm planning to do some more experimentation on this.

Another way to evaluate the latent space is to explore around the encoding around the points that the trained data is mapped to. To do this, I took several images, encoded them, found the closest distinct neighbor (using KNN) and then extracted images along the line between these two points (using a cosine squared parameter). Finally, I animited these images into a collection of GIFs (see below).

![image here]()

For images that have a similar neighbor the model seems
