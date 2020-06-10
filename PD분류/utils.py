import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import numpy as np
import random


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def str2bool(x):
    return x.lower() in ('true')

def normalize(X_train, X_test):

    mean = np.mean(X_train, axis=(0,1,2,3))
    std = np.std(X_train, axis=(0,1,2,3))

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test

def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] +2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0,0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad, mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh+crop_shape[0], nw:nw+crop_shape[1]]

    return batch

def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)): #n자리 bit를 랜덤생성 반환값은 10진수(e.g. 3을 입력했을시 0~7까지 값 반환, 1입력시 0,1 반환)
            batch[i] = np.fliplr(batch[i])
    return batch


def data_augmentation(batch, h_img_size, w_img_size):

    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch=batch, crop_shape = [h_img_size, w_img_size], padding=4)

    return batch