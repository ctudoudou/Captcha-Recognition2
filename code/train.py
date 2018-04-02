#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/2 上午9:54
# @Author  : tudoudou
# @File    : train.py
# @Software: PyCharm

from TFtools import TFRecord
from model import Net
from dataset import read_tfrecord, dic_
from keras.models import load_model
from PIL import Image
import numpy as np


def train():
    app = Net()
    app = app.create_model()
    images, labels = read_tfrecord()
    images_val, labels_val = read_tfrecord(type_='val')

    app.fit(images, labels, 128, 100, validation_data=(images_val, labels_val))

    app.save('../model/net.h5')


if __name__ == '__main__':
    do=input('train or predict: ')
    # train()
    if do not in ['train','predict']:
        raise ValueError('do must be train or predict !')

    if do=='train':
        train()
    else:
        boxs = [(5, 1, 17, 21), (17, 1, 29, 21), (29, 1, 41, 21), (41, 1, 53, 21)]
        app = load_model('../model/net.h5')
        while True:
            name = input("filename: ")
            img = Image.open(name).convert('L').convert('1')
            name = ''
            for x in range(len(boxs)):
                aaa = []
                roi = img.crop(boxs[x])
                roi = np.array(roi)
                aaa.append([roi])
                aaa = np.array(aaa)
                aaa = app.predict(aaa)
                name += dic_[np.argmax(aaa)]
            print(name)

