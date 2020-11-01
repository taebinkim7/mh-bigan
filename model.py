import tensorflow as tf
from tensorflow.keras import layers

# aae
def Encoder(lat_dim, hid_dim):
    inputs = tf.keras.Input(shape=[32, 32, 3])
    x = layers.Reshape((3072,))(inputs)
    x = layers.Dense(hid_dim)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(hid_dim)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(lat_dim)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def Decoder(lat_dim, hid_dim):
    inputs = tf.keras.Input(shape=(lat_dim,))
    x = layers.Dense(hid_dim)(inputs)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(hid_dim)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(3072, activation='tanh')(x)
    outputs = layers.Reshape([32, 32, 3])(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def D_aae(lat_dim, hid_dim):
    inputs = tf.keras.Input(shape=(lat_dim,))
    x = layers.Dense(hid_dim)(inputs)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(hid_dim)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# mc  
# def Transition(latent_dim):
#     inputs_x = tf.keras.Input(shape=[28, 28, 1])
#     inputs_e = tf.keras.Input(shape=(latent_dim,))

#     e = layers.Dense(7*7*256, use_bias=False)(inputs_e)
#     e = layers.BatchNormalization()(e)
#     e = layers.LeakyReLU()(e)
#     e = layers.Reshape([7, 7, 256])(e)
#     e = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(e)
#     e = layers.BatchNormalization()(e)
#     e = layers.LeakyReLU()(e)
#     e = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(e)
#     e = layers.BatchNormalization()(e)
#     e = layers.LeakyReLU()(e)
#     e = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False)(e)
#     e = layers.BatchNormalization()(e)
#     e = layers.LeakyReLU()(e)

#     x = layers.concatenate([inputs_x, e], axis=-1)
#     x = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU()(x)
#     x = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU()(x)
#     outputs = layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh')(x)
    
#     model = tf.keras.Model(inputs=[inputs_x, inputs_e], outputs=outputs)
#     return model

def Transition(latent_dim):
    inputs_x = tf.keras.Input(shape=[32, 32, 3])
    inputs_e = tf.keras.Input(shape=(latent_dim,))

    # e = layers.Dense(7*7*256, use_bias=False)(inputs_e)
    # e = layers.BatchNormalization()(e)
    # e = layers.LeakyReLU()(e)
    # e = layers.Reshape([7, 7, 256])(e)
    # e = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(e)
    # e = layers.BatchNormalization()(e)
    # e = layers.LeakyReLU()(e)
    # e = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(e)
    # e = layers.BatchNormalization()(e)
    # e = layers.LeakyReLU()(e)
    # e = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False)(e)
    # e = layers.BatchNormalization()(e)
    # e = layers.LeakyReLU()(e)

    e = layers.RepeatVector(1024)(inputs_e)
    e = layers.Reshape([32, 32, latent_dim])(e)
                       
    x = layers.concatenate([inputs_x, e])

    c1 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Dropout(0.2)(c3)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    c3 = layers.BatchNormalization()(c3)

    u2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c3)
    u2 = layers.concatenate([u2, c2])
    c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u2)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Dropout(0.1)(c4)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    c4 = layers.BatchNormalization()(c4)

    u1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u1 = layers.concatenate([u1, c1])
    c5 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u1)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Dropout(0.1)(c5)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    c5 = layers.BatchNormalization()(c5)

    outputs = layers.Conv2D(1, (1, 1), activation='tanh') (c5)

    model = tf.keras.Model(inputs=[inputs_x, inputs_e], outputs=outputs)
    return model

def D_image():
    inputs = tf.keras.Input(shape=[32, 32, 3])
    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(inputs)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# def D_mix_x():
#     inputs = tf.keras.Input(shape=[32, 32, 6])
#     x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(inputs)
#     x = layers.LeakyReLU()(x)
#     x = layers.Dropout(0.3)(x)
#     x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
#     x = layers.LeakyReLU()(x)
#     x = layers.Dropout(0.3)(x)
#     x = layers.Flatten()(x)
#     outputs = layers.Dense(1, activation='sigmoid')(x)
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     return model
