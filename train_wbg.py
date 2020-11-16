from argparse import ArgumentParser
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import os
from os import makedirs
import time
from IPython import display

from model import Encoder, Generator, Critic
from loss import W_loss
from util import plot_images, mh_update, gradient_penalty

parser = ArgumentParser(description='bigan')
parser.add_argument('--out_dir', type=str, action='store')
args = parser.parse_args()

# Hyperparameters
BUFFER_SIZE = 50000
BATCH_SIZE = 64
EPOCHS = 100
LATENT_DIM = 200
GP_WEIGHT = 10
NUM_CRITIC = 5

SIGMA = 1.
GAMMA = 0.1

NUM_EXAMPLES = 20
NUM_CHANNELS = 3

# Create a directory
makedirs(args.out_dir, exist_ok=True)

# # Load data
(train_images, train_labels), (_, _) = tf.keras.datasets.cifar10.load_data()
train_images = train_images.astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]

# (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
# train_images = np.pad(train_images, [(0,0), (2,2), (2,2)])
# train_images = train_images.reshape(train_images.shape[0], 32, 32, 1).astype('float32')
# train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]

# Define models
enc = Encoder(train_images[0].shape, LATENT_DIM)
gen = Generator(train_images[0].shape, LATENT_DIM)
crit = Critic(train_images[0].shape, LATENT_DIM)

# Define optimizers
eg_optimizer = tf.keras.optimizers.Adam(1e-4, 0.5, 0.9)
c_optimizer = tf.keras.optimizers.Adam(1e-4, 0.5, 0.9)

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Checkpoint
# wbg_ckpt_dir = './wbg_checkpoints'
wbg_ckpt_dir = os.path.join(args.out_dir, 'wbg_checkpoints')
wbg_ckpt_prefix = os.path.join(wbg_ckpt_dir, 'wbg_ckpt')
wbg_ckpt = tf.train.Checkpoint(step=tf.Variable(1), eg_optimizer=eg_optimizer, c_optimizer=c_optimizer,
                               enc=enc, gen=gen, crit=crit)
wbg_manager = tf.train.CheckpointManager(wbg_ckpt, wbg_ckpt_dir, max_to_keep=1)

# Train steps
@tf.function
def train_step_c(batch_x):
    with tf.GradientTape() as eg_tape, tf.GradientTape() as c_tape:
        x = batch_x
        ex = enc(x, training=True)
        
        z = tf.random.normal([x.shape[0], LATENT_DIM])
        gz = gen(z, training=True)
        
        x_ex = crit([x, ex], training=True)
        gz_z = crit([gz, z], training=True)
        
        gp = gradient_penalty(partial(crit, training=True), x, ex, z, gz)
        
        c_loss = W_loss(x_ex, gz_z) + GP_WEIGHT * gp

    c_gradient = c_tape.gradient(c_loss, crit.trainable_variables)
    c_optimizer.apply_gradients(zip(c_gradient, crit.trainable_variables))
    
    return c_loss

def train_step_eg(batch_x):
    with tf.GradientTape() as eg_tape:
        x = batch_x
        ex = enc(x, training=True)
        
        z = tf.random.normal([x.shape[0], LATENT_DIM])
        gz = gen(z, training=True)
        
        x_ex = crit([x, ex], training=True)
        gz_z = crit([gz, z], training=True)
        
        eg_loss = - W_loss(x_ex, gz_z)
        
    eg_gradient = eg_tape.gradient(eg_loss, enc.trainable_variables + gen.trainable_variables)
    eg_optimizer.apply_gradients(zip(eg_gradient, enc.trainable_variables + gen.trainable_variables))
    
    return eg_loss
  
def train(dataset, n_epoch):    
    wbg_ckpt.restore(wbg_manager.latest_checkpoint)
  
    for epoch in range(n_epoch):
        start = time.time()
        
        eg_loss, c_loss = 0, 0
        
        for batch in dataset:
          
            for _ in range(NUM_CRITIC):
                c_loss_batch = train_step_c(batch)
            
            eg_loss_batch = train_step_eg(batch)
            
            eg_loss += eg_loss_batch
            c_loss += c_loss_batch
        
        wbg_ckpt.step.assign_add(1)
        wbg_manager.save()
        
        display.clear_output(wait=True)
        
        x0 = train_images[0:NUM_EXAMPLES]
        ex0 = enc(x0, training=False)
        gex0 = gen(ex0, training=False)
        plot_images(int(wbg_ckpt.step), x0, gex0, args.out_dir, 'reconstruct')
        
        tf.random.set_seed(10)
        z = tf.random.normal([NUM_EXAMPLES, LATENT_DIM])
        gz = gen(z, training=False)
        ex_mh = tf.scan(mh_update, GAMMA * tf.ones(10), ex0)[-1]
        gex_mh = gen(ex_mh, training=False)
        plot_images(int(wbg_ckpt.step), gz, gex_mh, args.out_dir, 'generate_mh')
        
        print ('Time for epoch {} is {} sec'.format(int(wbg_ckpt.step), time.time()-start))
        print ('G loss is {} and D loss is {}'.format(eg_loss, c_loss))

# Train
train(train_dataset, EPOCHS)
# wbg_ckpt.save(file_prefix = wbg_ckpt_prefix)
