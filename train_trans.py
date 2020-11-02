import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import os
from os import makedirs
from tensorflow.keras import layers
import time

from model import Transition, D_image
from loss import D_loss, G_loss, Trans_loss
from func import mh_update, plot_images

from train_aae import encoder, decoder, d_aae, ae_optimizer, enc_optimizer, d_aae_optimizer, aae_ckpt_dir, aae_checkpoint, train_images

parser = ArgumentParser(description='TRANS')
parser.add_argument('--out_dir', type=str, action='store')
args = parser.parse_args()

aae_checkpoint.restore(tf.train.latest_checkpoint(aae_ckpt_dir))

# Hyperparameters
BUFFER_SIZE = 50000
BATCH_SIZE = 128
EPOCHS = 100
LATENT_DIM = 200
HIDDEN_DIM = 3000

SIGMA = 1.
GAMMA = 0.1

NUM_EXAMPLES = 20

# Create a directory
makedirs(args.out_dir, exist_ok=True)

# Define models
transition = Transition(LATENT_DIM)
d_image = D_image()

# Define optimizers
trans_optimizer = tf.keras.optimizers.Adam(1e-4)
d_image_optimizer = tf.keras.optimizers.Adam(1e-4)

# Load data

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_dataset1 = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Checkpoint
trans_ckpt_dir = './trans_checkpoints'
trans_ckpt_prefix = os.path.join(trans_ckpt_dir, "trans_ckpt")
trans_checkpoint = tf.train.Checkpoint(trans_optimizer=trans_optimizer, d_image_optimizer=d_image_optimizer,
                                       transition=transition, d_image=d_image)

# Train steps
@tf.function
def train_step_mc(batch_x, batch_x1):
    with tf.GradientTape() as trans_tape, tf.GradientTape() as d_image_tape:

        start_x = batch_x
        start_z = encoder(start_x, training=False)
        next_z = mh_update(start_z, GAMMA)
        e = next_z - start_z
        next_x = transition([start_x, e], training=True)
        next_ex = encoder(next_x, training=False)
        
        # Image loss
        real_out_image = d_image(batch_x1, training=True) # different shuffle
        fake_out_image = d_image(next_x, training=True)

        d_image_loss = D_loss(real_out_image, fake_out_image) 

        # Trans loss
        t_image_loss = G_loss(fake_out_image) 
        t_code_loss = mse(next_z, next_ex)
        trans_loss = t_image_loss + 100 * t_code_loss
    
    trans_gradient = trans_tape.gradient(trans_loss, transition.trainable_variables)
    trans_optimizer.apply_gradients(zip(trans_gradient, transition.trainable_variables))
    d_image_gradient = d_image_tape.gradient(d_image_loss, d_image.trainable_variables)
    d_image_optimizer.apply_gradients(zip(d_image_gradient, d_image.trainable_variables))

    return d_image_loss, trans_loss

def train_mc(dataset, dataset1, n_epoch):
    for epoch in range(n_epoch):
        tf.random.set_seed(10)
        start = time.time()

        d_image_loss, trans_loss = 0, 0

        for image_batch, image_batch1 in zip(dataset, dataset1):
            d_image_loss_batch, trans_loss_batch = train_step_mc(image_batch, image_batch1, image_batch2)
            d_image_loss += d_image_loss_batch 
            trans_loss += trans_loss_batch
        
        seed_images = train_images[0:NUM_EXAMPLES]
        e = GAMMA * tf.random.normal([NUM_EXAMPLES, LATENT_DIM])
        next_images = transition([seed_images, e], training=False)
        plot_images(epoch + 1, seed_images, next_images, args.out_dir)

        print('D loss is {} and T loss is {}'.format(d_image_loss, trans_loss))
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

# Train
train_mc(train_dataset, train_dataset1, EPOCHS)
trans_checkpoint.save(file_prefix = trans_ckpt_prefix)
