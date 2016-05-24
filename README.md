Learned Similarity Autoencoder for Modelling and Reconstructing Video Frames
==============================

Tesorflow implementation of [Autoencoding beyond pixels using a learned similarity metric](http://arxiv.org/abs/1512.09300). 

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

Put the dataset directory in a directory called 'datasets' within the code project file.

To turn a video into frames and frame into video you can use [ffmpeg](https://ffmpeg.org/)


