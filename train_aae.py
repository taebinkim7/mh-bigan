from argparse import ArgumentParser
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import os
from os import makedirs
import PIL
from tensorflow.keras import layers
import time
from IPython import display

parser = ArgumentParser(description='MCMC_GAN')
parser.add_argument('--out_dir', type=str, action='store')
args = parser.parse_args()

makedirs(args.out_dir, exist_ok=True)
