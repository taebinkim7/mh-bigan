from argparse import ArgumentParser
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import os
from os import makedirs
from tensorflow.keras import layers
import time

from model import Encoder, Generator, Discriminator
from loss import D_loss, G_loss
from func import plot_images

parser = ArgumentParser(description='AAE')
parser.add_argument('--out_dir', type=str, action='store')
args = parser.parse_args()

# Hyperparameters
BUFFER_SIZE = 50000
BATCH_SIZE = 128
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
enc_optimizer = tf.keras.optimizers.Adam(1e-4)
gen_optimizer = tf.keras.optimizers.Adam(1e-4)
disc_optimizer = tf.keras.optimizers.Adam(1e-4)

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Checkpoint
bg_ckpt_dir = './bg_checkpoints'
bg_ckpt_prefix = os.path.join(bg_ckpt_dir, "bg_ckpt")
bg_ckpt = tf.train.Checkpoint(enc_optimizer=enc_optimizer, gen_optimizer=gen_optimizer, disc_optimizer=disc_optimizer,
                              enc=enc, gen=gen, disc=disc)


# Train steps
@tf.function
def train_step_aae(batch_x):
    with tf.GradientTape() as ae_tape, tf.GradientTape() as enc_tape, tf.GradientTape() as d_aae_tape:
        # Autoencoder / Encoder
        fake_z = encoder(batch_x, training=True)
        fake_x = decoder(fake_z, training=True)

        # Autoencoder loss
        ae_loss = AE_loss(batch_x, fake_x)

        # Discriminator
        real_z = tf.random.normal([batch_x.shape[0], LATENT_DIM], mean=0., stddev=SIGMA)

        real_output = d_aae(real_z, training=True)
        fake_output = d_aae(fake_z, training=True)

        # Encoder loss
        enc_loss = G_loss(fake_output)

        # Discriminator Loss
        d_aae_loss = D_loss(real_output, fake_output)

    ae_gradient = ae_tape.gradient(ae_loss, encoder.trainable_variables + decoder.trainable_variables)
    ae_optimizer.apply_gradients(zip(ae_gradient, encoder.trainable_variables + decoder.trainable_variables))
    enc_gradient = enc_tape.gradient(enc_loss, encoder.trainable_variables)
    enc_optimizer.apply_gradients(zip(enc_gradient, encoder.trainable_variables))
    d_aae_gradient = d_aae_tape.gradient(d_aae_loss, d_aae.trainable_variables)
    d_aae_optimizer.apply_gradients(zip(d_aae_gradient, d_aae.trainable_variables))

    return ae_loss, enc_loss, d_aae_loss

def train_aae(dataset, n_epoch):    
    for epoch in range(n_epoch):
        start = time.time()

        for image_batch in dataset:
            train_step_aae(image_batch)
        
        seed_images = train_images[0:NUM_EXAMPLES]
        next_images = decoder(encoder(seed_images, training=False), training=False)
        plot_images(epoch + 1, seed_images, next_images, args.out_dir)
        
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

# Train
train_aae(train_dataset, EPOCHS)
aae_checkpoint.save(file_prefix = aae_ckpt_prefix)
