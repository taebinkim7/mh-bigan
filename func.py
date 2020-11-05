import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

def mh_update(prev, gam, sig=1):
        cand = prev + tf.random.normal(prev.shape, mean=0.0, stddev=gam)
        p = tf.minimum(1.0, tf.math.exp(tf.reduce_sum(prev**2 - cand**2, axis=1) / sig**2)) # tf.minumum unnecessary
        u = tf.random.uniform(p.shape)
        return tf.where(tf.expand_dims(u < p, axis=1), cand, prev)

def plot_images(epoch, sample_input, sample_next, out_dir, img_title):
    n_examples = sample_input.shape[0]
    fig = plt.figure(figsize=(n_examples, 2))
    for j in range(n_examples):
        plt.subplot(2, n_examples, j + 1)
#         plt.imshow(tf.squeeze(sample_input[j]) / 2 + .5)
        plt.imshow(tf.squeeze(sample_input[j]), cmap='gray')
        plt.axis('off')  

        plt.subplot(2, n_examples, n_examples + j + 1)
#         plt.imshow(tf.squeeze(sample_next[j]) / 2 + .5)
        plt.imshow(tf.squeeze(sample_next[j]), cmap='gray')
        plt.axis('off')   

    plt.savefig(os.path.join(out_dir, img_title + '_at_epoch_{:04d}.png'.format(epoch)))
    plt.close(fig)   

def gradient_penalty(f, real, fake):
    epsilon = tf.random.uniform([real.shape[0], 1, 1, 1])
    diff = fake - real
    inter = real + (epsilon * diff)
    with tf.GradientTape() as t_tape:
        t_tape.watch(inter)
        pred = f(inter)
    grad = t_tape.gradient(pred, [inter])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
    gp = tf.reduce_mean((slopes - 1.)**2)
    return gp
