"""

This is the associated code for the project 'Autoencoding Video Frames' for Terence Broad Msci Creative Computing Dissertation project.

This code is heavily modified and extended from the DCGAN implementation by Taehoon Kim:  https://github.com/carpedm20/DCGAN-tensorflow

His model was modified to be an autoenocer, render large non-square images, and the learn using a learned similarity metric. 

The structure of the code in this file is derived from his code but has been completely re-structured and re-purposed.

This project is an impelementation of 'Autoencoding beyond pixels using a learned similarity metric' by Larsen et al. (2015). 

Here is a link to the paper: http://arxiv.org/pdf/1512.09300v1.pdf

"""

import os
import time
from glob import glob
import tensorflow as tf

from ops import *
from utils import *
from random import shuffle

class Autoencoder(object):
    def __init__(self, sess, noise=1,image_size=256, is_crop=True,
                 batch_size=12, sample_size = 12, image_shape=[144, 256, 3],
                 y_dim=None, z_dim=200, ef_dim = 80, gf_dim= 80, df_dim=80, 
                 c_dim=3, gamma = 2.5, theta = 1, dataset_name='default',
                 checkpoint_dir=None):
        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of decoder filters in first conv layer. [80]
            df_dim: (optional) Dimension of discrimator filters in first conv layer. [80]
            ef_dim: (optional) Dimension of encodr filters in first conv layer. [80]
   
            c_dim: (optional) Dimension of image color. [3]
        """
        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.image_shape = image_shape
        self.gamma = gamma
        self.theta = theta
        self.noise_var = noise

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.ef_dim = ef_dim

        self.c_dim = 3


        #BATCH NORMALISATION
        self.e_bn1 = batch_norm(batch_size, name='e_bn1')
        self.e_bn2 = batch_norm(batch_size, name='e_bn2')
        self.e_bn3 = batch_norm(batch_size, name='e_bn3')

        self.g_bn0 = batch_norm(batch_size, name='g_bn0')
        self.g_bn1 = batch_norm(batch_size, name='g_bn1')
        self.g_bn2 = batch_norm(batch_size, name='g_bn2')
        self.g_bn3 = batch_norm(batch_size, name='g_bn3')

        self.d_bn1 = batch_norm(batch_size, name='d_bn1')
        self.d_bn2 = batch_norm(batch_size, name='d_bn2')
        self.d_bn3 = batch_norm(batch_size, name='d_bn3')

        self.kl_bn = batch_norm(batch_size, name='kl_bn')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

        
    def build_model(self):

        #SAMPLE REAL IMAGES
        self.images = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape, name='real_images')
        self.sample_images= tf.placeholder(tf.float32, [self.sample_size] + self.image_shape, name='sample_images')
        #ADJUSTABLE NOISE
        self.noise_var = tf.placeholder(tf.float32, [], name = 'noise_var')

        #APPORXIMATE POSTERIOR LATENT REPRESENTATION WITH ENCODER 
        self.z_mu, self.z_log_sigma = self.encoder(self.images)

        #FEED REAL DATA SAMPLE TO DISCRIMINATOR
        self.D_real,self.D_real_h4, self.D_real_h3, self.D_real_h2, self.D_real_h1, self.D_real_h0 = self.discriminator(self.images)

        t_vars = tf.trainable_variables()

        #SAMPLE NOISE
        eps = tf.random_normal((self.batch_size, self.z_dim), 0, tf.clip_by_value(self.noise_var,1e-10,1.0), dtype=tf.float32)

        #CALCULATE LATENT REPRESENTATION Z AS A DETERMINISTIC FUNCTION OF NOISE
        self.z = tf.add(self.z_mu, tf.mul(tf.sqrt(tf.exp(self.z_log_sigma)), eps))

        self.mean_eps = tf.reduce_mean(tf.reduce_mean(eps,0),0)

        #CALCULATE KL DIVERGENCE OBJECTIVE FUNCTION OF APRROXIMATE POSTERION Z TO PRIOR NORMAL DISTRIBUTION 
        self.kl_div = (- 0.5 * tf.reduce_mean((tf.reduce_mean(1 + tf.clip_by_value(self.z_log_sigma,-5.0,5.0) - tf.square(tf.clip_by_value(self.z_mu,-5.0,5.0)) - tf.exp(tf.clip_by_value(self.z_log_sigma,-5.0,5.0)), 1))))
        
        #GENERATE RECONSTRUCTED SAMPLES
        self.G = self.decoder(self.z)

        self.pz = tf.placeholder(tf.float32, [None, self.z_dim],
                                name='pz')

        #GENERATE RANDOM SAMPLES
        self.G_pz = self.decoder(self.pz,reuse = True)
        
        #FEED RANDOM AND RECONSTRUCTED SAMPLES TO DISCRIMINATOR
        self.D_fake, self.D_fake_h4, self.D_fake_h3, self.D_fake_h2, self.D_fake_h1, self.D_fake_h0  = self.discriminator(self.G, reuse=True)
        self.D_pz, self.D_pz_real_h4, self.D_pz_real_h3, self.D_pz_real_h2, self.D_pz_real_h1, self.D_pz_real_h0  = self.discriminator(self.G_pz, reuse=True)

        #BINARY CROSS ENTROPY OF REAL/GENERATED DISCRIMINATION
        self.d_loss_real = binary_cross_entropy_with_logits(tf.ones_like(self.D_real), tf.clip_by_value(self.D_real, 1e-10,1.0))/self.batch_size
        self.d_loss_fake = binary_cross_entropy_with_logits(tf.zeros_like(self.D_fake), tf.clip_by_value(self.D_fake, 1e-10,1.0))/self.batch_size
        self.d_loss_pz = binary_cross_entropy_with_logits(tf.zeros_like(self.D_pz), tf.clip_by_value(self.D_pz, 1e-10,1.0))/self.batch_size

        #GAN LOSS OBJECTIVE FUNCTION
        self.gan_loss = (self.d_loss_real + self.d_loss_pz + self.d_loss_fake)/6.0

        #CALCULATE LEARNED SIMILIARITY OBJECTIVE FUNCTION
        self.likeness = (tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.reduce_mean((tf.square(self.D_fake_h3 - self.D_real_h3)),0),0),0),0) 
         + tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.reduce_mean((tf.square(self.D_fake_h2 - self.D_real_h2)),0),0),0),0)
         + tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.reduce_mean((tf.square(self.D_fake_h1 - self.D_real_h1)),0),0),0),0)
         + tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.reduce_mean((tf.square(self.D_fake_h0 - self.D_real_h0)),0),0),0),0)
            )/4.0

        #RESPECTIVE ERROR GRADIENTS
        self.e_loss =  tf.clip_by_value((self.kl_div + self.likeness), 1e-10, 100)
        self.g_loss =  tf.clip_by_value((self.gamma*self.likeness - self.gan_loss), -100, 100)
        self.d_loss =  tf.clip_by_value(self.gan_loss, 1e-10, 100)

        #SUMMARIES
        self.kl_sum = tf.histogram_summary("kl_sum", self.kl_div)
        self.likeness_sum = tf.histogram_summary("likeness_sum", self.likeness)
        self.gan_loss_sum = tf.histogram_summary("gan_loss_sum", self.gan_loss)
        self.e_loss_sum = tf.histogram_summary("e_loss_sum", self.e_loss)
        self.g_loss_sum = tf.histogram_summary("g_loss_sum", self.g_loss)
        self.d_loss_sum = tf.histogram_summary("d_loss_sum", self.d_loss)
        self.G_sum = tf.image_summary("G", self.G)
        
        #TRAINABLE VBARIABLES
        self.t_vars = tf.trainable_variables()
        self.g_vars = [var for var in t_vars if 'e_' in var.name]
        self.e_vars = [var for var in t_vars if 'g_' in var.name]
        self.d_vars = [var for var in t_vars if 'd_' in var.name]

        self.saver = tf.train.Saver()   
    
    def train(self, config):
        """Train AUTOENCODER"""
        data = glob(os.path.join("./datasets", config.dataset, "*.png"))

        #RESPECTIVE OPTIMIZERS
        e_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.e_loss, var_list=self.e_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list= self.d_vars)
        
        tf.initialize_all_variables().run()

        self.saver = tf.train.Saver()
        self.e_sum = tf.merge_summary([self.kl_sum, self.e_loss_sum])
        self.g_sum = tf.merge_summary([self.g_loss_sum])
        self.d_sum = tf.merge_summary([self.d_loss_sum, self.gan_loss_sum, self.likeness_sum])

        self.writer = tf.train.SummaryWriter("./logs", self.sess.graph_def)

        #SAMPLES IMAGES AND RANDOM LATENT REPRESENTATIONS
        sample_z = np.random.uniform(-1, 1, size=(self.sample_size , self.z_dim))
        sample_files = data[0:self.sample_size]
        sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)
        samples = np.array(sample).astype(np.float32)
        counter = 1
        start_time = time.time()
        noise = config.noise

        #LOAD PRETRAINED MODEL
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        #CYCLE THROUGH DATASET
        for epoch in xrange(config.epoch):
            data = glob(os.path.join("./datasets", config.dataset, "*.png"))
            shuffle(data)
            batch_idxs = min(len(data), config.train_size)/config.batch_size

            for idx in xrange(0, batch_idxs):
                batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
                batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                            .astype(np.float32)

                #TRAIN AND SAMPLE INPUT BATCH AND RECONSTRUCTION
                if np.mod(counter, 25) == 1:
                    samples, _, __, ___, summary_e, summary_g, summary_d  = self.sess.run([self.G, e_optim, g_optim, d_optim, self.e_sum, self.g_sum, self.d_sum],
                        feed_dict={ self.images: batch_images, self.pz: batch_z, self.noise_var: noise} )
                    sample_images = samples

                    save_images(samples, [3, 4],'./samples/%s_%s_output.png' % (epoch, idx))
                    save_images(batch_images, [3, 4],'./samples/%s_%s_input.png' % (epoch, idx))

                #TRAIN AND SAMPLE RANDOMLY GENERATED SAMPLES
                elif np.mod(counter+1, 100) == 1:
                    samples,samples_fake, _, __, ___, summary_e, summary_g, summary_d  = self.sess.run([self.G_pz, self.G_pz, e_optim, g_optim, d_optim, self.e_sum, self.g_sum, self.d_sum],
                        feed_dict={ self.images: batch_images, self.pz: batch_z, self.noise_var: noise} )
                    sample_images = samples

                    save_images(samples, [3, 4],'./samples/%s_%s_random.png' % (epoch, idx))

                #TRAIN WITHOUT SAMPLING AND EXPORTING IMAGES
                else:
                     _, __, ___, summary_e, summary_g, summary_d  = self.sess.run([e_optim, g_optim, d_optim, self.e_sum, self.g_sum, self.d_sum],
                        feed_dict={ self.images: batch_images, self.pz: batch_z, self.noise_var: noise} )
                
                #SUMMARIES
                self.writer.add_summary(summary_d, counter)
                error_e = self.e_loss.eval({ self.images: batch_images, self.pz: batch_z, self.noise_var: noise})
                error_g = self.g_loss.eval({ self.images: batch_images, self.pz: batch_z, self.noise_var: noise})
                error_d = self.d_loss.eval({ self.images: batch_images, self.pz: batch_z, self.noise_var: noise})
                loss_kl = self.kl_div.eval({ self.images: batch_images, self.pz: batch_z, self.noise_var: noise})
                loss_likeness = self.likeness.eval({ self.images: batch_images, self.pz: batch_z, self.noise_var: noise})
                mean_noise = self.mean_eps.eval({ self.images: batch_images, self.pz: batch_z, self.noise_var: noise})

                #SAVE MODEL
                if np.mod(counter, 500) == 2 and counter > 450:
                    self.save(config.checkpoint_dir, counter)

                #PRINT PROGRESS INFORMATION TO TERMINAL
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, encoder loss: %.8f, decoder loss: %.8f , discriminator loss: %.8f, kl_div:  %.8f, likeness:  %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, error_e, error_g, error_d, loss_kl, loss_likeness))
                
                counter += 1

    #RUN NETWORK WITHOUT TRAINING
    def run(self, config):
       """Train VAE"""
       data = glob(os.path.join("./datasets", config.dataset, "*.png"))
       
       tf.initialize_all_variables().run()

       self.saver = tf.train.Saver()
       self.e_sum = tf.merge_summary([self.kl_sum, self.e_loss_sum])
       self.g_sum = tf.merge_summary([self.g_loss_sum])
       self.d_sum = tf.merge_summary([self.d_loss_sum, self.gan_loss_sum, self.likeness_sum])

       self.writer = tf.train.SummaryWriter("./logs", self.sess.graph_def)

       sample_z = np.random.uniform(-1, 1, size=(self.sample_size , self.z_dim))
       sample_files = data[0:self.sample_size]
       sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop) for sample_file in sample_files]
       sample_images = np.array(sample).astype(np.float32)
       samples = np.array(sample).astype(np.float32)
       counter = 1
       start_time = time.time()
       
       if config.noise != 1:
            noise = config.noise
       else:
            noise = 0


       if self.load(self.checkpoint_dir):
           print(" [*] Load SUCCESS")
       else:
           print(" [!] Load failed...")

       
       data = sorted(glob(os.path.join("./datasets", config.dataset, "*.png")), key=numericalSort)
       batch_idxs = min(len(data), config.train_size)/config.batch_size

       for idx in xrange(0, batch_idxs):
           batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
           batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop) for batch_file in batch_files]
           batch_images = np.array(batch).astype(np.float32)

           batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                       .astype(np.float32)

           
           counter += 1


           samples, summary_e, summary_g, summary_d = self.sess.run([self.G, self.e_sum, self.g_sum, self.d_sum],
               feed_dict={ self.images: batch_images, self.pz: batch_z, self.noise_var: noise} ) 
           sample_images = samples

           save_frames(sample_images, self.batch_size, idx,'./output/output_')

           loss_kl = self.kl_div.eval({ self.images: batch_images, self.pz: batch_z, self.noise_var: noise})
           loss_likeness = self.likeness.eval({ self.images: batch_images, self.pz: batch_z, self.noise_var: noise})
           mean_noise = self.mean_eps.eval({ self.images: batch_images, self.pz: batch_z, self.noise_var: noise})

           
           print("Outputing Batch [%6d/%6d] time_taken: %4.4f, frame_num: [%8d], time_in_film (mins): %6.4f, kl_div:  %.8f, likeness:  %.8f" \
               % (idx, batch_idxs, time.time() - start_time, idx*self.batch_size, (self.batch_size/1440.0)*(idx+1), loss_kl, loss_likeness))



    #ENCODER NETWORK
    def encoder(self, image, reuse=False, y=None):

        if reuse:
            tf.get_variable_scope().reuse_variables()

        h0 = lrelu(conv2d(image, self.ef_dim, name='e_h0_conv'))
        h1 = lrelu(self.e_bn1(conv2d(h0, self.ef_dim*2, name='e_h1_conv')))
        h2 = lrelu(self.e_bn2(conv2d(h1, self.ef_dim*4, name='e_h2_conv')))
        h3 = lrelu(self.e_bn3(conv2d(h2, self.ef_dim*8, name='e_h3_conv')))
        h4_ = tf.reshape(h3,[self.batch_size, self.ef_dim*8*9*16])
        z_log_sigma = 0.5 * tf.nn.tanh(linear(h4_, self.z_dim, 'e_z_log_sigma'))
        z_mu = tf.nn.tanh(linear(h4_, self.z_dim, 'e_z_mu'))


        return (z_mu,z_log_sigma)

    #DECODER NETWORK
    def decoder(self, z, reuse=False, y=None):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        # project `z` and reshape
        self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*9*16, 'g_h0_lin', with_w=True)

        self.h0 = tf.reshape(self.z_, [-1, 9, 16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(self.h0))

        self.h1, self.h1_w, self.h1_b = deconv2d(h0, 
            [self.batch_size, 18, 32, self.gf_dim*4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(self.h1))

        h2, self.h2_w, self.h2_b = deconv2d(h1,
            [self.batch_size, 36, 64, self.gf_dim*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3, self.h3_w, self.h3_b = deconv2d(h2,
            [self.batch_size, 72, 128, self.gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4, self.h4_w, self.h4_b = deconv2d(h3,
            [self.batch_size, 144, 256, 3], name='g_h4', with_w=True)

        return tf.nn.tanh(h4)

    #DISCRIMINATOR NETWORK
    def discriminator(self, image, reuse=False, y=None):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

        return (tf.nn.sigmoid(h4),h4,h3,h2,h1,h0)

    #SAVE MODEL
    def save(self, checkpoint_dir, step):
        model_name = "VAE.model"
        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    #LOAD MODEL
    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
