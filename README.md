# Adversarial-Autoencoder-Experiments

## Summary

This repository has a series of experiments using some of the adversarial autoencoders described in [Adversarial Autoencoders](https://arxiv.org/pdf/1511.05644.pdf); I also found parts of the [Wasserstein Auto-encoders](https://arxiv.org/pdf/1711.01558.pdf) very useful. The experiments include

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



