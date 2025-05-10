import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D, Input, concatenate, BatchNormalization, LeakyReLU
)
from tensorflow.keras.initializers import RandomNormal

def downscale(num_filters):
    block = tf.keras.Sequential()
    block.add(Conv2D(num_filters, kernel_size=4, strides=2, padding='same', 
                    kernel_initializer='he_normal', use_bias=False))
    block.add(LeakyReLU(alpha=0.2))
    block.add(BatchNormalization())
    return block

def Discriminator():
    image = Input(shape=(256,256,3), name="ImageInput")
    target = Input(shape=(256,256,3), name="TargetInput")
    x = concatenate([image, target])

    x = downscale(64)(x)
    x = downscale(128)(x)
    x = downscale(512)(x)

    initializer = RandomNormal(stddev=0.02, seed=42)

    x = Conv2D(512, kernel_size=4, strides=1, padding='same', 
              kernel_initializer=initializer, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(1, kernel_size=4, padding='same', 
              kernel_initializer=initializer)(x)

    discriminator = Model(inputs=[image, target], outputs=x, name="Discriminator")
    return discriminator 