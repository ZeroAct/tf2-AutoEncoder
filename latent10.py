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

### Encoder 정의
encoder_input = Input(shape=(32,32,3))

x = Conv2D(32, 3, strides=1, padding='same')(encoder_input)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Conv2D(64, 3, strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Conv2D(64, 3, strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Conv2D(64, 3, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Flatten()(x)

encoder_output = Dense(10)(x)

encoder = Model(encoder_input, encoder_output)

### Decoder 정의
decoder_input = Input(shape=(10,))

x = Dense(8*8*64)(decoder_input)
x = Reshape((8, 8, 64))(x)

x = Conv2DTranspose(64, 3, strides=1, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Conv2DTranspose(64, 3, strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Conv2DTranspose(64, 3, strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Conv2DTranspose(32, 3, strides=1, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

decoder_output = Conv2DTranspose(3, 3, strides=1, padding='same', activation='tanh')(x)

decoder = Model(decoder_input, decoder_output)

### Hyperparameter 정의
LEARNING_RATE = 0.00001
BATCH_SIZE = 32

### AutoEncoder 정의
encoder_in = Input(shape=(32,32,3))
x = encoder(encoder_in)
decoder_out = decoder(x)

auto_encoder = Model(encoder_in, decoder_out)

### 학습
auto_encoder.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                     loss=tf.keras.losses.MeanSquaredError())

model_dir = './latent10/'
os.mkdir(model_dir)
model_path = model_dir + '{epoch}__loss_{loss}.h5'

checkpoint_callback = ModelCheckpoint(model_path,
                                      save_best_only=True,
                                      save_weights_only=False,
                                      monitor='loss',
                                      verbose=1)
earlystop_callback = EarlyStopping(monitor='loss',
                                   patience=2)

auto_encoder.fit(x_train, x_train,
                 batch_size=BATCH_SIZE,
                 epochs=100,
                 callbacks=[checkpoint_callback, earlystop_callback])

import matplotlib.pyplot as plt

decoded_images = auto_encoder.predict(x_train[:15])

fig, axes = plt.subplots(3, 5)
fig.set_size_inches(12, 6)
for i in range(15):
    t = x_train[i]
    t = (t-t.min())/(t.max()-t.min())*255
    axes[i//5, i%5].imshow(t.astype(np.uint8))
    axes[i//5, i%5].axis('off')
plt.tight_layout()
plt.title('Original Images')
plt.show()

fig, axes = plt.subplots(3, 5)
fig.set_size_inches(12, 6)
for i in range(15):
    t = decoded_images[i]
    t = (t-t.min())/(t.max()-t.min())*255
    axes[i//5, i%5].imshow(t.astype(np.uint8))
    axes[i//5, i%5].axis('off')
plt.tight_layout()
plt.title('Auto Encoder Images')
plt.show()