#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/2 上午9:53
# @Author  : tudoudou
# @File    : dataset.py
# @Software: PyCharm


import os
from TFtools import TFRecord
import numpy as np
from PIL import Image
import tensorflow as tf
import time

dic = {'9': 2, 'f': 15, 'z': 9, 'o': 16, '7': 4, '5': 33, '8': 11, 'w': 35, '0': 3, 'y': 32, 'k': 29, 'b': 28, 'n': 6,
       'r': 0, 'j': 19, 's': 13, 'i': 5, '3': 26, 'x': 25, 'u': 1, 'a': 14, 't': 12, 'p': 34, '6': 8, 'q': 23, 'h': 17,
       'd': 21, '1': 10, 'v': 22, 'g': 7, '4': 24, '2': 31, 'c': 20, 'l': 30, 'e': 27, 'm': 18}

dic_ = {0: 'r', 1: 'u', 2: '9', 3: '0', 4: '7', 5: 'i', 6: 'n', 7: 'g', 8: '6', 9: 'z', 10: '1', 11: '8', 12: 't',
        13: 's', 14: 'a', 15: 'f', 16: 'o', 17: 'h', 18: 'm', 19: 'j', 20: 'c', 21: 'd', 22: 'v', 23: 'q', 24: '4',
        25: 'x', 26: '3', 27: 'e', 28: 'b', 29: 'k', 30: 'l', 31: '2', 32: 'y', 33: '5', 34: 'p', 35: 'w'}


def read_tfrecord(tfr=None, type_='train', num=1500):
    if tfr == None:
        tfr = TFRecord({'img': [bytes], 'labels': [int] * 36})
    if type_ == 'train':
        num = 1500
        example = tfr.reader('./tfrecord/*.tfrecord')
    else:
        num = 100
        example = tfr.reader('./tfrecord/*.tfrecords')
    img = tf.decode_raw(example['img'], tf.uint8)
    img = tf.reshape(img, [20, 12])
    lab = example['labels']
    images, labels = [], []

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(num):
            res1, res2 = sess.run([img, lab])

            # res1 *= 255
            # Image.fromarray(res1).show()
            # print(dic_[np.argmax(res2)])
            images.append([res1])
            labels.append(res2)

        coord.request_stop()
        coord.join(threads)

        return np.array(images), np.array(labels)


def write_tfrecord(tfr):
    writer = tfr.writer('./tfrecord/', pre_file_capacity=500)
    boxs = [(5, 1, 17, 21), (17, 1, 29, 21), (29, 1, 41, 21), (41, 1, 53, 21)]
    lab = np.zeros((1, 36))
    for parent, dirnames, filenames in os.walk('./data_biaoji'):
        for i in filenames:
            if len(i) != 8:
                print(i)
                raise ValueError
            img = Image.open(os.path.join(parent, i)).convert('L').convert('1')

            for x in range(len(boxs)):
                roi = img.crop(boxs[x])
                roi = np.array(roi).reshape((1, 240))
                lab = np.zeros((1, 36))
                lab[0][dic[i[x]]] = 1
                lab = lab.astype(np.int)
                writer.add_example({'img': [roi.astype(np.uint8).tostring()], 'labels': lab[0]})
    writer.close()


if __name__ == '__main__':
    try:
        os.mkdir('./tfrecord')
    except FileExistsError:
        pass
    tfr = TFRecord({'img': [bytes], 'labels': [int] * 36})
    write_tfrecord(tfr)
    os.rename('./tfrecord/2.tfrecord', './tfrecord/2.tfrecords')
