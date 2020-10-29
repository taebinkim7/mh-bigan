import tensorflow as tf

cross_entropy = tf.keras.losses.BinaryCrossentropy()
mse = tf.keras.losses.MeanSquaredError()

def D_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def G_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def AE_loss(input, output):
    return mse(input, output)

def Trans_loss(fake_out_image, fake_out_x, fake_out_z):
    image_loss = cross_entropy(tf.ones_like(fake_out_image), fake_out_image)
    mix_loss = mse(fake_out_z, fake_out_x)
    total_loss = image_loss + mix_loss
    return total_loss
