"""

This is the associated code for the project 'Autoencoding Video Frames' for Terence Broad Msci Creative Computing Dissertation project.

This code is heavily modified and extended from the DCGAN implementation by Taehoon Kim:  https://github.com/carpedm20/DCGAN-tensorflow

His model was modified to be an autoenocer, render large non-square images, and the learn using a learned similarity metric. 

Therefore some of the function were written by him (especially in ops.py and utils.py) but model.py is almost completely re-written.

This project is an impelementation of 'Autoencoding beyond pixels using a learned similarity metric' by Larsen et al. (2015). 

Here is a link to the paper: http://arxiv.org/pdf/1512.09300v1.pdf

"""






import os
import numpy as np
import tensorflow as tf

from model import Autoencoder
from utils import pp, visualize, to_json

flags = tf.app.flags
flags.DEFINE_integer("epoch", 5, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("noise", 1, "Amount of noise placed on latent representation between 0-1. For intial training this should be 1, can be reduced in later stages of training (fine tuning), and should be 0 when running autoencoder after training")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 12, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 108, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_string("dataset", "bergen2", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("output_dir", "output", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_run", False, "True for running, False for training [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)


    config = tf.ConfigProto(
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9),
        device_count = {'GPU': 1},
        allow_soft_placement=True
        #log_device_placement=True,
    )
    config.device_filters.append('/gpu:0')
    config.device_filters.append('/cpu:0')

    with tf.Session(config=config) as sess:
        #with tf.device('/gpu:0'):

        autoencoder = Autoencoder(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size,
                dataset_name=FLAGS.dataset, noise = FLAGS.noise, is_crop=FLAGS.is_crop, checkpoint_dir=FLAGS.checkpoint_dir)

        if FLAGS.is_train:
            autoencoder.train(FLAGS)
        elif FLAGS.is_run:
            autoencoder.run(FLAGS)
        else:
            autoencoder.load(FLAGS.checkpoint_dir)

"""
        to_json("./web/js/layers.js", [autoencoder.h0_w, autoencoder.h0_b, autoencoder.g_bn0],
                                      [autoencoder.h1_w, autoencoder.h1_b, autoencoder.g_bn1],
                                      [autoencoder.h2_w, autoencoder.h2_b, autoencoder.g_bn2],
                                      [autoencoder.h3_w, autoencoder.h3_b, autoencoder.g_bn3],
                                      [autoencoder.h4_w, autoencoder.h4_b, None])

        # Below is codes for visualization
        OPTION = 2
        visualize(sess, autoencoder, FLAGS, OPTION)
"""

if __name__ == '__main__':
    tf.app.run()
