from argparse import ArgumentParser
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import os
from os import makedirs
from tensorflow.keras import layers
import time

from model import Encoder, Decoder, D_aae
from loss import D_loss, G_loss, AE_loss

parser = ArgumentParser(description='AAE')
parser.add_argument('--out_dir', type=str, action='store')
args = parser.parse_args()

# Hyperparameters
BUFFER_SIZE = 60000
BATCH_SIZE = 128
EPOCHS = 300
LATENT_DIM = 30
HIDDEN_DIM = 2000
SIGMA = 1.
NUM_EXAMPLES = 20

makedirs(args.out_dir, exist_ok=True)

# Load data
(train_images, train_labels), (_, _) = tf.keras.datasets.cifar10.load_data()
train_images = train_images.astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]

# Define models
encoder = Encoder(LATENT_DIM, HIDDEN_DIM)
decoder = Decoder(LATENT_DIM, HIDDEN_DIM)
d_aae = D_aae(LATENT_DIM, HIDDEN_DIM)

# Define optimizers
ae_optimizer = tf.keras.optimizers.Adam(1e-4)
enc_optimizer = tf.keras.optimizers.Adam(1e-4)
d_aae_optimizer = tf.keras.optimizers.Adam(1e-4)

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Checkpoint
aae_ckpt_dir = './aae_checkpoints'
aae_ckpt_prefix = os.path.join(aae_ckpt_dir, "aae_ckpt")
aae_checkpoint = tf.train.Checkpoint(ae_optimizer=ae_optimizer, enc_optimizer=enc_optimizer, d_aae_optimizer=d_aae_optimizer,
                                 encoder=encoder, decoder=decoder, d_aae=d_aae)


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
    seed_images = train_images[0:NUM_EXAMPLES]
    next_images = decoder(encoder(seed_images, training=False), training=False)
    
    for epoch in range(n_epoch):
        start = time.time()

        for image_batch in dataset:
            train_step_aae(image_batch)
        
        # Plot images
        fig = plt.figure(figsize=(NUM_EXAMPLES, 2))
        for i in range(NUM_EXAMPLES):
            plt.subplot(2, n_examples, i + 1)
            plt.imshow(tf.squeeze(seed_images[i]) / 2 + .5)
            plt.axis('off')
            plt.subplot(2, n_examples, n_examples + i + 1)
            plt.imshow(tf.squeeze(next_images[i]) / 2 + .5)
            plt.axis('off')      
        plt.savefig(os.path.join(args.out_dir, 'image_at_epoch_{:04d}.png'.format(epoch)))
        plt.close(fig) 
        
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

# Train
train_aae(train_dataset, EPOCHS)
aae_checkpoint.save(file_prefix = aae_ckpt_prefix)
