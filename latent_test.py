### 필요한 모듈 임포트
import os

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Dropout, BatchNormalization, Reshape, LeakyReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from tensorflow.keras import Input

import tensorflow as tf
import numpy as np

print(tf.test.is_gpu_available())

### 데이터 로드
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_valid, y_valid) = cifar10.load_data()

### 데이터 프로세싱
x_train = x_train.reshape(-1, 32, 32, 3)
x_train = x_train / 127.5 - 1
x_valid = x_valid.reshape(-1, 32, 32, 3)
x_valid = x_valid / 127.5 - 1

auto_encoder = tf.keras.models.load_model('./denoise_latent512/final.h5')

encoder = Model(Input(shape=(32,32,3)), auto_encoder.layers[1].outputs[0])
