import tensorflow as tf
from tensorflow.keras import layers

def Encoder(img_dim, lat_dim):
    inputs = tf.keras.Input(shape=img_dim)

    x = layers.Conv2D(128, (5, 5), (2, 2), padding='same', kernel_initializer='he_normal')(inputs)
    # x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, (5, 5), (2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(512, (5, 5), (2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(512, (5, 5), (2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Flatten()(x)

    outputs = layers.Dense(lat_dim)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def Generator(img_dim, lat_dim):
    dim16 = img_dim[0] // 16

    inputs = tf.keras.Input(shape=(lat_dim,))

    x = layers.Dense(512 * dim16 * dim16, use_bias=False)(inputs)
    x = layers.Reshape([dim16, dim16, 512])(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(512, (5, 5), (2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(256, (5, 5), (2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(128, (5, 5), (2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    outputs = layers.Conv2DTranspose(3, (5, 5), (2, 2), padding='same', activation='tanh', kernel_initializer='he_normal')(x) ####MNIST
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def Discriminator(img_dim, lat_dim):
    inputs_x = tf.keras.Input(shape=img_dim)
    inputs_z = tf.keras.Input(shape=(lat_dim,))

    x = layers.Conv2D(128, (5, 5), (2, 2), padding='same', kernel_initializer='he_normal')(inputs_x)
    # x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, (5, 5), (2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(512, (5, 5), (2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(512, (5, 5), (2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)

    z = layers.Flatten()(inputs_z)
    z = layers.Dense(512)(z)
    z = layers.Dropout(0.2)(z)

    xz = layers.concatenate([x, z])
    xz = layers.Dense(1024)(xz)
    xz = layers.LeakyReLU()(xz)

    outputs = layers.Dense(1)(xz)
    model = tf.keras.Model(inputs=[inputs_x, inputs_z], outputs=outputs)
    return model

def Critic(img_dim, lat_dim):
    inputs_x = tf.keras.Input(shape=img_dim)
    inputs_z = tf.keras.Input(shape=(lat_dim,))
    
    z = layers.Dense(img_dim[0]*img_dim[1])(inputs_z)
    z = layers.LeakyReLU()(z)
    z = layers.Reshape([img_dim[0], img_dim[1], 1])(z)

#     z = layers.RepeatVector(1024)(inputs_z)
#     z = layers.Reshape([32, 32, lat_dim])(z)
    
    x = layers.concatenate([inputs_x, z])

    x = layers.Conv2D(128, (5, 5), (2, 2), padding='same', kernel_initializer='he_normal')(x)
    # x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, (5, 5), (2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
#     x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(512, (5, 5), (2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
#     x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(512, (5, 5), (2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
#     x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(1)(x)
    model = tf.keras.Model(inputs=[inputs_x, inputs_z], outputs=outputs)
    return model
