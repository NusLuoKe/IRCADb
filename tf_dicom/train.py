#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/12 19:51
# @File    : train.py
# @Author  : NUS_LuoKe

from tf_dicom.load_dicom import *

base_dir = "F:/IRCAD/3Dircadb1/"

slice_path_list, liver_path_list = get_slice_liver_path(base_dir, shuffle=True)
training_set, validation_set, test_set = get_tra_val_test_set(slice_path_list, liver_path_list)

'''
for batch_x_y in get_batch(slice_path="", liver_path="", batch_size=4):
    batch_x = batch_x_y[0]
    batch_y = batch_x_y[1]
    break
'''