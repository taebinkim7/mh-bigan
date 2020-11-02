import tensorflow as tf
from tensorflow.keras import layers

# aae
def Encoder(lat_dim, hid_dim):
    inputs = tf.keras.Input(shape=[32, 32, 3])
    x = layers.Reshape((3072,))(inputs)
    x = layers.Dense(hid_dim)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(hid_dim // 2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(lat_dim)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def Decoder(lat_dim, hid_dim):
    inputs = tf.keras.Input(shape=(lat_dim,))
    x = layers.Dense(hid_dim // 2)(inputs)
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
    x = layers.Dense(hid_dim // 2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# mc  
def Transition(latent_dim):
    inputs_x = tf.keras.Input(shape=[32, 32, 3])
    inputs_e = tf.keras.Input(shape=(latent_dim,))

    # e = layers.RepeatVector(1024)(inputs_e)
    # e = layers.Reshape([1, 1, latent_dim])(inputs_e)
    e = layers.Dense(4*4*256, use_bias=False)(inputs_e)
    e = layers.BatchNormalization()(e)
    e = layers.LeakyReLU()(e)
    e = layers.Reshape([4, 4, 256])(e)
                       
    e1 = layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)(e)
    e1 = layers.BatchNormalization()(e1)
    e1 = layers.LeakyReLU()(e1)

    e2 = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(e1)
    e2 = layers.BatchNormalization()(e2)
    e2 = layers.LeakyReLU()(e2)

    e3 = layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False)(e2)
    e3 = layers.BatchNormalization()(e3)
    e3 = layers.LeakyReLU()(e3)

    # x = layers.concatenate([inputs_x, e])

    c1 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs_x) # [32, 32, 32]
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    c1 = layers.BatchNormalization()(c1)

    p2 = layers.MaxPooling2D((2, 2))(c1) # [16, 16, 32]
    c2 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    c2 = layers.BatchNormalization()(c2)

    p3 = layers.MaxPooling2D((2, 2))(c2) # [8, 8, 64]
    c3 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Dropout(0.2)(c3)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    c3 = layers.BatchNormalization()(c3)
    
    p4 = layers.MaxPooling2D((2, 2))(c3) # [4, 4, 128]
    c4 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Dropout(0.2)(c4)
    c4 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    c4 = layers.BatchNormalization()(c4)

    
    # xe = layers.concatenate([xe, e])
    ce = layers.add([c4, e])
    # xe = layers.Dense(256 * 4 * 4)(xe)
    # xe = layers.BatchNormalization()(xe)

    # u4 = layers.Reshape([4, 4, 256])(xe)

    u5 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(ce)
    # ce1 = layers.add([c3, e1])
    # u5 = layers.concatenate([u5, ce1])
    u5 = layers.add([u5, e1])
    c5 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Dropout(0.1)(c5)
    c5 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    c5 = layers.BatchNormalization()(c5)

    u6 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
    # ce2 = layers.add([c2, e2])
    # u6 = layers.concatenate([u6, ce2])
    u6 = layers.add([u6, e2])
    c6 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Dropout(0.1)(c6)
    c6 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    c6 = layers.BatchNormalization()(c6)

    u7 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
    # ce3 = layers.add([c1, e3])
    # u7 = layers.concatenate([u7, ce3])
    u7 = layers.add([u7, e3])
    c7 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Dropout(0.1)(c7)
    c7 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    c7 = layers.BatchNormalization()(c7)

    outputs = layers.Conv2D(3, (1, 1), activation='tanh')(c7)

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
