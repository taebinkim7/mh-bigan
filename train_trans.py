import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import os
from os import makedirs
from tensorflow.keras import layers
import time

from model import Encoder, Decoder, D_aae
from loss import D_loss, G_loss, AE_loss

from train_aae import encoder, decoder, d_aae, ae_optimizer, enc_optimizer, d_aae_optimizer

parser = ArgumentParser(description='TRANS')
parser.add_argument('--out_dir', type=str, action='store')
args = parser.parse_args()

makedirs(args.out_dir, exist_ok=True)

aae_ckpt_dir = './aae_checkpoints'
aae_ckpt_prefix = os.path.join(aae_ckpt_dir, "aae_ckpt")
aae_checkpoint = tf.train.Checkpoint(ae_optimizer=ae_optimizer, enc_optimizer=enc_optimizer, d_aae_optimizer=d_aae_optimizer,
                                     encoder=encoder, decoder=decoder, d_aae=d_aae)




plot_images(no n_steps)
