#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/2 上午9:52
# @Author  : tudoudou
# @File    : get_picture.py
# @Software: PyCharm

import os
from urllib.request import urlretrieve


def save_img(img_url, file_name, file_path='data'):
    try:
        if not os.path.exists(file_path):
            print('文案夾', file_path, '無法找到，重新創建')
            os.makedirs(file_path)

        filename = '{}{}{}{}'.format(file_path, os.sep, file_name, '.gif')

        urlretrieve(img_url, filename=filename)
    except IOError as e:
        print('文案操作失敗', e)
    except Exception as e:
        print('error ：', e)


if __name__ == '__main__':
    i = 0
    while (i < 100):
        save_img('http://jwsys.ctbu.edu.cn/CheckCode.aspx', i)
        i += 1
    print('end !')
