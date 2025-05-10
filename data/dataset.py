import tensorflow as tf
import numpy as np
from glob import glob
import os

IMG_SIZE = (256, 256)
BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE

def load_and_preprocess_image(rgb_path, ir_path):
    rgb_image = tf.io.read_file(rgb_path)
    ir_image = tf.io.read_file(ir_path)
    rgb_image = tf.image.decode_jpeg(rgb_image, channels=3)
    ir_image = tf.image.decode_jpeg(ir_image, channels=3)
    rgb_image = tf.image.resize(rgb_image, IMG_SIZE, method='bicubic')
    ir_image = tf.image.resize(ir_image, IMG_SIZE, method='bicubic')
    rgb_image = rgb_image / 255.0
    ir_image = ir_image / 255.0
    return rgb_image, ir_image

def create_dataset(rgb_image_path, ir_image_path, frame_ids, allData=False):
    if allData:
        fids = frame_ids
    else:
        fids = frame_ids
    rgb_paths = [f'{rgb_image_path}/254p RGB Frame ({fid}).jpg' for fid in fids]
    ir_paths = [f'{ir_image_path}/254p Thermal Frame ({fid}).jpg' for fid in fids]
    dataset = tf.data.Dataset.from_tensor_slices((rgb_paths, ir_paths))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    return dataset

def prepare_datasets(dataset, train_size, val_size):
    dataset = dataset.shuffle(buffer_size=len(dataset), reshuffle_each_iteration=True)
    train_ds = dataset.take(train_size).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    remaining = dataset.skip(train_size)
    val_ds = remaining.take(val_size).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    test_ds = remaining.skip(val_size).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return train_ds, val_ds, test_ds 