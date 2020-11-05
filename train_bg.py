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
from loss import D_loss, G_loss
from func import plot_images

parser = ArgumentParser(description='bigan')
parser.add_argument('--out_dir', type=str, action='store')
args = parser.parse_args()

# Hyperparameters
BUFFER_SIZE = 50000
BATCH_SIZE = 64
EPOCHS = 50
LATENT_DIM = 128
# HIDDEN_DIM = 3000

# SIGMA = 1.
# GAMMA = 0.1

NUM_EXAMPLES = 20

# Create a directory
makedirs(args.out_dir, exist_ok=True)

# Load data
(train_images, train_labels), (_, _) = tf.keras.datasets.cifar10.load_data()
train_images = train_images.astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]

# Define models
enc = Encoder(train_images[0].shape, LATENT_DIM)
gen = Generator(train_images[0].shape, LATENT_DIM)
disc = Discriminator(train_images[0].shape, LATENT_DIM)

# Define optimizers
g_optimizer = tf.keras.optimizers.Adam(1e-4)
d_optimizer = tf.keras.optimizers.Adam(1e-4)

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Checkpoint
bg_ckpt_dir = './bg_checkpoints'
bg_ckpt_prefix = os.path.join(bg_ckpt_dir, "bg_ckpt")
bg_ckpt = tf.train.Checkpoint(g_optimizer=g_optimizer, d_optimizer=d_optimizer,
                              enc=enc, gen=gen, disc=disc)


# Train steps
@tf.function
def train_step(batch_x):
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        x = batch_x
        ex = enc(x, training=True)
        
        z = tf.random.normal([x.shape[0], LATENT_DIM])
        gz = gen(z, training=True)
        
        x_ex = disc([x, ex], training=True)
        gz_z = disc([gz, z], training=True)
        
        d_loss = D_loss(x_ex, gz_z)
        g_loss = G_loss(x_ex, gz_z)

    g_gradient = g_tape.gradient(g_loss, enc.trainable_variables + gen.trainable_variables)
    g_optimizer.apply_gradients(zip(g_gradient, enc.trainable_variables + gen.trainable_variables))
    d_gradient = d_tape.gradient(d_loss, disc.trainable_variables)
    d_optimizer.apply_gradients(zip(d_gradient, disc.trainable_variables))
    
    return g_loss, d_loss

def train(dataset, n_epoch):    
    for epoch in range(n_epoch):
        start = time.time()
        
        g_loss, d_loss = 0, 0
        
        for batch in dataset:
            g_loss_batch, d_loss_batch = train_step(batch)
            g_loss += g_loss_batch
            d_loss += d_loss_batch
        
        display.clear_output(wait=True)
        seed_images = train_images[0:NUM_EXAMPLES]
        next_images = gen(enc(seed_images, training=False), training=False)
        plot_images(epoch + 1, seed_images, next_images, args.out_dir)
        
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        print ('G loss is {} and D loss is {}.format(g_loss, d_loss))

# Train
train(train_dataset, EPOCHS)
bg_checkpoint.save(file_prefix = bg_ckpt_prefix)
