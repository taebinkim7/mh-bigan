import tensorflow as tf
from tensorflow.keras import layers
def Encoder(img_dim, lat_dim):
    inputs = tf.keras.Input(shape=img_dim)

    x = layers.Conv2D(64, (5, 5), (2, 2), padding='same', kernel_initializer='he_normal')(inputs)
    # x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, (5, 5), (2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, (5, 5), (2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(512, (5, 5), (2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Flatten()(x)

    outputs = layers.Dense(lat_dim, kernel_initializer='he_normal')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def Generator(img_dim, lat_dim):
    dim16 = img_dim[0] // 16

    inputs = tf.keras.Input(shape=(lat_dim,))

    x = layers.Dense(512 * dim16 * dim16, use_bias=False)(inputs)
    x = layers.Reshape([dim16, dim16, 512])(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(256, (5, 5), (2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(128, (5, 5), (2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(64, (5, 5), (2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    outputs = layers.Conv2DTranspose(1, (5, 5), (2, 2), padding='same', activation='tanh', kernel_initializer='he_normal')(x) ####MNIST
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def Discriminator(img_dim, lat_dim):
    inputs_x = tf.keras.Input(shape=img_dim)
    inputs_z = tf.keras.Input(shape=(lat_dim,))

    x = layers.Conv2D(64, (5, 5), (2, 2), padding='same', kernel_initializer='he_normal')(inputs_x)
    # x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, (5, 5), (2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, (5, 5), (2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(512, (5, 5), (2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)

    z = layers.Flatten()(inputs_z)
    z = layers.Dense(512, kernel_initializer='he_normal')(z)
    z = layers.Dropout(0.2)(z)

    xz = layers.concatenate([x, z])

    outputs = layers.Dense(1, kernel_initializer='he_normal')(xz)
    model = tf.keras.Model(inputs=[inputs_x, inputs_z], outputs=outputs)
    return model
