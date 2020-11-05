import tensorflow as tf

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def D_loss(x_ex, gz_z):
    real_loss = cross_entropy(tf.ones_like(x_ex), x_ex)
    fake_loss = cross_entropy(tf.zeros_like(gz_z), gz_z)
    total_loss = real_loss + fake_loss
    return total_loss

def EG_loss(x_ex, gz_z):
    real_loss = cross_entropy(tf.zeros_like(x_ex), x_ex)
    fake_loss = cross_entropy(tf.ones_like(gz_z), gz_z)
    total_loss = real_loss + fake_loss
    return total_loss

def W_loss(x_ex, gz_z):
    real_loss = - tf.reduce_mean(x_ex)
    fake_loss = tf.reduce_mean(gz_z)
    total_loss = real_loss + fake_loss
    return total_loss
