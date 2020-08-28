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

encoder = auto_encoder.layers[1]
decoder = auto_encoder.layers[2]

test_img = x_train[7].copy()
noise_img = test_img + 0.1*np.random.randn(32,32,3)

latent = encoder.predict(np.array([noise_img]))
result = decoder.predict(latent)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2)
fig.set_size_inches(12, 6)

t = test_img
t = (t-t.min())/(t.max()-t.min())*255
axes[0].imshow(t.astype(np.uint8))
axes[0].axis('off')

t = result[0]
t = (t-t.min())/(t.max()-t.min())*255
axes[1].imshow(t.astype(np.uint8))
axes[1].axis('off')

plt.tight_layout()
plt.show()