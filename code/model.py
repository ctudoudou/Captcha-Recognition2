#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/2 上午9:53
# @Author  : tudoudou
# @File    : model.py
# @Software: PyCharm


import tensorflow as tf
import keras
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import BatchNormalization, Activation
from keras.layers import Dense, AveragePooling2D, Flatten
from keras.layers import Dropout
from keras.models import model_from_json, load_model, Model


class Net():

    def __init__(self):
        print('init ')

    def my_model(self):
        inputs = Input(shape=(1, 20, 12))

        x = Conv2D(
            filters=16, kernel_size=(2, 4), padding='same', name='Conv1', data_format='channels_first')(inputs)
        x = BatchNormalization(axis=1, name='BN_Conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), data_format='channels_first')(x)

        x = Conv2D(
            filters=4, kernel_size=(2, 2), padding='same', name='Conv1', data_format='channels_first')(inputs)
        x = BatchNormalization(axis=1, name='BN_Conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), data_format='channels_first')(x)

        x = AveragePooling2D((2, 2), name='avg_pool')(x)
        x = Flatten()(x)
        x = Dropout(0.2)(x)
        x = Dense(36, activation='softmax', name='sotfmax36')(x)

        model = Model(inputs, x, name='My_Resnet')

        return model

    def create_model(self):
        model = self.my_model()
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model


if __name__ == '__main__':
    app = Net()
    app = app.create_model()

    import numpy as np

    X = np.random.randint(1, 100, [100, 1, 20, 12])
    Y = np.random.randint(1, 36, [100, 36])

    print(X)
    print(Y)

    app.fit(X, Y, 64, 5)
