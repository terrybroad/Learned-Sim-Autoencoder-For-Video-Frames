Learned Similarity Autoencoder for Modelling and Reconstructing Video Frames
==============================

Tesorflow implementation of [Autoencoding beyond pixels using a learned similarity metric](http://arxiv.org/abs/1512.09300). 

This was written for Tensorflow version 0.7 which is no longer supported. Unfortunately training a new model using Tensorflow 0.8 or later does not work as a model will learn at all. I believe this is an issue with the loss function of the discriminator and how it is backpropagated but I have been unable to debug this problem. If someone does fix it please submit a pull request.

A lot of the architecture is derived from this codebase [DCGAN-Tensorflow](https://github.com/carpedm20/DCGAN-tensorflow).

This project is designed to read 256x144 png's and that are indexed in numerial order.

This project was implemented for my project on reconstructing videos with neural networks - [read more](http://terencebroad.com/autoencodingbladerunner.html)


To train a model with a dataset:

    $ python main.py --dataset DATASETNAME --is_train True 

You may want to adjust the amount of noise injected into the latent space:

    $ python main.py --dataset DATASETNAME --is_train True --noise 0.5

This parameter controls the standard deviation of noise epsilon from mean of 0.

The output frames in sequence using an exisiting model:

    $ python main.py --dataset DATASETNAME --is_run True 

In this version of the code 1000 minibatches must be processed before a model is saved, and therefore can be used to output frames sequentailly. This can be edited in line 245 of model.py.

Put the dataset directory in a directory called 'datasets' within the code project file.

To turn a video into frames and frame into video you can use [ffmpeg](https://ffmpeg.org/)


