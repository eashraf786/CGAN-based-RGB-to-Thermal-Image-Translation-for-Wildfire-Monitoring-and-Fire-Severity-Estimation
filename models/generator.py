import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, Input, Conv2DTranspose, Dropout, 
    concatenate, BatchNormalization, LeakyReLU, ReLU
)
from tensorflow.keras.initializers import RandomNormal

def downscale(num_filters):
    block = Sequential()
    block.add(Conv2D(num_filters, kernel_size=4, strides=2, padding='same', 
                    kernel_initializer='he_normal', use_bias=False))
    block.add(LeakyReLU(alpha=0.2))
    block.add(BatchNormalization())
    return block

def upscale(num_filters, use_dropout=False):
    block = Sequential()
    block.add(Conv2DTranspose(num_filters, kernel_size=4, strides=2, padding='same', 
                            kernel_initializer='he_normal', use_bias=False))
    block.add(LeakyReLU(alpha=0.2))
    block.add(BatchNormalization())
    if use_dropout:
        block.add(Dropout(0.5))
    block.add(ReLU())
    return block

def Generator():
    inputs = Input(shape=(256,256,3), name="InputLayer")

    encoder = [
        downscale(64),
        downscale(128),
        downscale(256),
        downscale(512),
        downscale(512),
        downscale(512),
        downscale(512),
    ]

    latent_space = downscale(512)

    decoder = [
        upscale(512, use_dropout=True),  
        upscale(512, use_dropout=True),
        upscale(512),
        upscale(512),
        upscale(256),
        upscale(128),
        upscale(64),
    ]

    x = inputs
    skips = []
    for layer in encoder:
        x = layer(x)
        skips.append(x)

    x = latent_space(x)

    skips = reversed(skips)
    for up, skip in zip(decoder, skips):
        x = up(x)
        x = concatenate([x, skip])

    initializer = RandomNormal(stddev=0.02, seed=42)
    outputs = Conv2DTranspose(3, kernel_size=4, strides=2, 
                            kernel_initializer=initializer, 
                            activation='tanh', 
                            padding='same')(x)

    generator = Model(inputs=inputs, outputs=outputs, name="Generator")
    return generator 