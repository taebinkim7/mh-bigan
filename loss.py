import tensorflow as tf

cross_entropy = tf.keras.losses.BinaryCrossentropy()

def D_loss(x_ex, gz_z):
    real_loss = cross_entropy(tf.ones_like(x_ex), x_ex)
    fake_loss = cross_entropy(tf.zeros_like(gz_z), gz_z)
    total_loss = real_loss + fake_loss
    return total_loss

def G_loss(x_ex, gz_z):
    real_loss = cross_entropy(tf.zeros_like(x_ex), x_ex)
    fake_loss = cross_entropy(tf.ones_like(gz_z), gz_z)
    total_loss = real_loss + fake_loss
    return total_loss
