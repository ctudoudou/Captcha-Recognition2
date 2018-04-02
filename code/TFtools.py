#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/31 下午10:07
# @Author  : wangqi,tudoudou
# @File    : TFtools.py
# @Software: PyCharm

import os
import numpy as np
from PIL import Image
import tensorflow as tf


class TFRecord(object):
    '''tfrecord文件读写工具，仅支持fixlen文件

    用法示例：
    ```
    import numpy as np
    tfr = TFRecord({'img': [bytes], 'labels': [int] * 2})
    # 数据写入
    writer = tfr.writer('/tmp', pre_file_capacity=5)
    for i in range(13):
        writer.add_example({'img': [np.ones([10, 5, 5]).tostring()], 'labels':[0, i]})
    writer.close()
    # 数据读取
    example = tfr.reader()
    ```
    '''

    def __init__(self, features_type):
        '''初始化TFRecord对象

        Args:
            features_type: 是一个dict。键值分别表示特征名称与对应类型，其类型是一个list
                例如：{'img': [bytes], 'label': [int] * 2}，
        '''
        self.features_type = features_type
        self.save_dir = None

    def writer(self, save_dir, pre_file_capacity=5000, prefix=None, suffix='tfrecord'):
        '''tfrecord文件写入

        Args:
            save_dir: tfrecord文件保存的文件夹地址
            pre_file_capacity: 每个tfrecord文件的容量，默认的每个文件可存储5000个样本，为`None`表示不限制单个文件的容量
            prefix: tfrecord文件的前缀名，可以为空
            suffix: tfrecord文件的后缀名，默认为tfrecord

        Returns:
            返回当前对象
        '''

        self.save_dir = save_dir
        self.pre_file_capacity = pre_file_capacity
        self.prefix = '' if prefix is None else prefix
        self.suffix = 'tfrecord' if suffix is None else suffix
        self._file = None
        self._file_idx = -1  # file索引 创建几个tfr文件就对应有几个file
        self._current_file_examples = 0  # 正在写入的tfr文件中的样本数量
        self.num_of_examples = 0  # 所有样本数量
        return self

    def _int64_feature(self, values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

    def _bytes_feature(self, values):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

    def _float_feature(self, values):
        return tf.train.Feature(float_list=tf.train.FloatList(value=values))

    def _open_file(self):
        '''创建一个tfrecords文件'''
        self._file_idx += 1
        path = os.path.join(self.save_dir, '%s%d.%s' % (self.prefix, self._file_idx, self.suffix))
        self._file = tf.python_io.TFRecordWriter(path)

    def _close_file(self):
        '''关闭正在开着的tfrecords文件流'''
        self._file.close()

    def close(self):
        self._close_file()

    def _features_dict(self, features):
        f_d = dict()
        for key, val in self.features_type.items():
            if val[0] is int:
                feature_fn = self._int64_feature
            elif val[0] is float:
                feature_fn = self._float_feature
            else:
                feature_fn = self._bytes_feature
            f_d[key] = feature_fn(features[key])
        return f_d

    def add_example(self, features):
        '''添加一个样本'''
        if self.save_dir is None:
            raise NameError('writer is not initializer!')
        if self._file is None:
            self._open_file()
        if self.pre_file_capacity is not None:
            if self._current_file_examples == self.pre_file_capacity:
                self._current_file_examples = 0
                self._close_file()
                self._open_file()

        features = self._features_dict(features)
        example = tf.train.Example(features=tf.train.Features(feature=features))
        self._file.write(example.SerializeToString())
        self._current_file_examples += 1
        self.num_of_examples += 1

    def reader(self, pattern=None, num_epochs=None):
        '''tfrecord文件读取

        Args:
            pattern: glob通配符
            num_epochs: 文件读取代数，默认为无限制

        Returns:
            返回一个样本对象
        '''
        if pattern is None and self.save_dir is not None:
            pattern = self.save_dir
        filenames = tf.train.match_filenames_once(pattern)
        filename_queue = tf.train.string_input_producer(filenames)
        reader = tf.TFRecordReader()
        key, value = reader.read(filename_queue)

        features = {}
        for k, v in self.features_type.items():
            shape = []
            if len(v) > 1:
                shape.append(len(v))
            if v[0] == int:
                dtype = tf.int64
            elif v[0] == float:
                dtype = tf.float32
            else:
                dtype = tf.string
            features[k] = tf.FixedLenFeature(shape, dtype)

        example = tf.parse_single_example(value, features=features)
        return example


def demo():
    import numpy as np
    tfr = TFRecord({'img': [bytes], 'labels': [int] * 2})
    writer = tfr.writer('/tmp/test', pre_file_capacity=5)
    for i in range(13):
        writer.add_example({'img': [np.ones([10, 5, 5]).astype(np.uint8).tostring()], 'labels': [0, i]})
    writer.close()

    example = tfr.reader('/tmp/test/*.tfrecord')
    img = tf.decode_raw(example['img'], tf.uint8)
    img = tf.reshape(img, [10, 5, 5])
    lab = example['labels']

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        res1, res2 = sess.run([img, lab])

        print(res1)
        print(res2)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    demo()
