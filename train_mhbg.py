from argparse import ArgumentParser
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import os
from os import makedirs
import time
from IPython import display

from model import Encoder, Generator, Discriminator
from loss import D_loss, EG_loss
from util import plot_images, mh_update

parser = ArgumentParser(description='bigan')
parser.add_argument('--out_dir', type=str, action='store')
args = parser.parse_args()

# Hyperparameters
BUFFER_SIZE = 50000
BATCH_SIZE = 64
EPOCHS = 100
LATENT_DIM = 128

SIGMA = 1.
GAMMA = 0.1

NUM_EXAMPLES = 20
NUM_CHANNELS = 1
NUM_STEPS = 5

# Create a directory
makedirs(args.out_dir, exist_ok=True)

# Load data
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
disc = Discriminator(train_images[0].shape, LATENT_DIM)

# Define optimizers
eg_optimizer = tf.keras.optimizers.Adam(1e-4)
d_optimizer = tf.keras.optimizers.Adam(1e-4)

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Checkpoint
# mhbg_ckpt_dir = './mhbg_checkpoints'
mhbg_ckpt_dir = os.path.join(args.out_dir, 'mhbg_checkpoints')
mhbg_ckpt_prefix = os.path.join(mhbg_ckpt_dir, 'mhbg_ckpt')
mhbg_ckpt = tf.train.Checkpoint(eg_optimizer=eg_optimizer, d_optimizer=d_optimizer,
                              enc=enc, gen=gen, disc=disc)


# Train steps
@tf.function
def train_step(batch_x, k):
    with tf.GradientTape() as eg_tape, tf.GradientTape() as d_tape:
        x = batch_x
        ex = enc(x, training=True)
        
        z = tf.random.normal([x.shape[0], LATENT_DIM])
        gz = gen(z, training=True)
        
#         ex1 = mh_update(ex, GAMMA)
#         x1 = gen(ex1, training=True)
        ex1 = tf.scan(mh_update, GAMMA * tf.ones(k), ex)[-1]
        x1 = gen(ex1, training=True)
        
        x_ex = disc([x, ex], training=True)
        gz_z = disc([gz, z], training=True)
        x1_ex1 = disc([x1, ex1], training=True)
        
        d_loss = D_loss(x_ex, gz_z) + D_loss(x_ex, x1_ex1)
        eg_loss = EG_loss(x_ex, x1_ex1) + 0.5 * EG_loss(x1_ex1, gz_z)

    eg_gradient = eg_tape.gradient(eg_loss, enc.trainable_variables + gen.trainable_variables)
    eg_optimizer.apply_gradients(zip(eg_gradient, enc.trainable_variables + gen.trainable_variables))
    d_gradient = d_tape.gradient(d_loss, disc.trainable_variables)
    d_optimizer.apply_gradients(zip(d_gradient, disc.trainable_variables))
    
    return eg_loss, d_loss

def train(dataset, n_epoch):    
    for epoch in range(n_epoch):
        start = time.time()
        
        eg_loss, d_loss = 0, 0
        
        k = np.random.choice(NUM_STEPS) + 1
        for batch in dataset:
            eg_loss_batch, d_loss_batch = train_step(batch, k)
            eg_loss += eg_loss_batch
            d_loss += d_loss_batch
        
        display.clear_output(wait=True)
        seed_images = train_images[0:NUM_EXAMPLES]
        next_images = gen(enc(seed_images, training=False), training=False)
        plot_images(epoch + 1, seed_images, next_images, args.out_dir, 'reconstruct')
        
        seed_codes = tf.random.normal([2*NUM_EXAMPLES, LATENT_DIM])
        fake_images = gen(seed_codes, training=False)
        plot_images(epoch + 1, fake_images[0:NUM_EXAMPLES], fake_images[NUM_EXAMPLES:2*NUM_EXAMPLES], args.out_dir, 'generate')
        
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        print ('G loss is {} and D loss is {}'.format(eg_loss, d_loss))

# Train
train(train_dataset, EPOCHS)
mhbg_ckpt.save(file_prefix = mhbg_ckpt_prefix)
