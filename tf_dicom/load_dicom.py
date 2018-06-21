#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/12 14:35
# @File    : load_dicom.py
# @Author  : NUS_LuoKe

import math
import os
import random

import numpy as np
import pydicom


def set_center_crop(x, width, height, center, row_index=0, col_index=1):
    h, w = x.shape[row_index], x.shape[col_index]
    center_h, center_w = center[row_index], center[col_index]

    if (h <= height) or (w <= width):
        raise AssertionError("The size of cropping should smaller than the original image")
    else:
        top_h = int(np.floor(center_h + height / 2.))
        bottom_h = int(np.floor(center_h - height / 2.))
        left_w = int(np.floor(center_w - width / 2.))
        right_w = int(np.floor(center_w + width / 2.))

        return x[bottom_h:top_h, left_w:right_w]


def crop(x, width, height, random_crop=False, row_index=0, col_index=1):
    '''
    :param x: numpy.array. An image with dimension of [row, col, channel] (default).
    :param width: Size of width.
    :param height: Size of height.
    :param random_crop: boolean, If True, randomly crop, else central crop.
    :param row_index: index of row.
    :param col_index: index of col.
    :return:  numpy.array. A processed image.
    '''

    h, w = x.shape[row_index], x.shape[col_index]

    if (h <= height) or (w <= width):
        raise AssertionError("The size of cropping should smaller than the original image")

    if random_crop:
        h_offset = int(np.random.uniform(0, h - height) - 1)
        w_offset = int(np.random.uniform(0, w - width) - 1)
        return x[h_offset:height + h_offset, w_offset:width + w_offset]
    else:  # central crop
        h_offset = int(np.floor((h - height) / 2.))
        w_offset = int(np.floor((w - width) / 2.))
        h_end = h_offset + height
        w_end = w_offset + width
        return x[h_offset:h_end, w_offset:w_end]


def get_slice_liver_path(base_dir, shuffle=True):
    slice_path_list = []  # 将所有图片的文件名的路径存成slice_path_list
    liver_path_list = []  # 将所有图片中liver的mask的文件名的路径存成liver_path_list
    # patient__num = len(os.listdir(base_dir))
    patient_id_list = [i for i in range(1, 20)]  # 留出最后一个病人的数据做验证集

    for patient_id in patient_id_list:
        patient_dicom_path = "3Dircadb1." + str(patient_id) + "/PATIENT_DICOM"
        liver_dicom_path = "3Dircadb1." + str(patient_id) + "/MASKS_DICOM/portalvein"
        slice_path = os.path.join(base_dir, patient_dicom_path)
        liver_path = os.path.join(base_dir, liver_dicom_path)

        for slice in os.listdir(slice_path):
            single_slice_path = os.path.join(slice_path, slice)
            slice_path_list.append(single_slice_path)

        for liver in os.listdir(liver_path):
            single_liver_path = os.path.join(liver_path, liver)
            liver_path_list.append(single_liver_path)

        if shuffle == True:
            rand_num = random.randint(0, 100)
            random.seed(rand_num)
            random.shuffle(slice_path_list)
            random.seed(rand_num)
            random.shuffle(liver_path_list)

    return slice_path_list, liver_path_list


def get_tra_val_set(slice_path, liver_path, tra_ratio=0.8):
    '''
    :param slice_path: 存放所有slice文件路径的list
    :param liver_path: 存放所有slice对应的liver文件路径的list
    :param tra_ratio: 训练集的比例
    :return: list type. eg: training_set = [[1], [2]], [1]和[2]为存放img和label的文件路径的list
    '''
    item_num = len(slice_path)
    train_num = int(item_num * tra_ratio)

    training_set_slice = slice_path[:train_num]
    training_set_liver = liver_path[:train_num]
    validation_set_slice = slice_path[train_num:]
    validation_set_liver = liver_path[train_num:]

    training_set = [training_set_slice, training_set_liver]
    validation_set = [validation_set_slice, validation_set_liver]
    return training_set, validation_set


def get_batch(slice_path, liver_path, batch_size, crop=False, center=None, height=None, width=None):
    batch_num = int(math.ceil(len(slice_path) / batch_size))

    for i in range(1, batch_num):
        if (batch_size * batch_num) == len(slice_path):
            slice_batch = slice_path[batch_size * (i - 1):batch_size * i]
            liver_batch = liver_path[batch_size * (i - 1):batch_size * i]

        elif (batch_size * batch_num) != len(slice_path):
            if i < batch_num:
                slice_batch = slice_path[batch_size * (i - 1):batch_size * i]
                liver_batch = liver_path[batch_size * (i - 1):batch_size * i]
            else:
                slice_batch = slice_path[batch_size * (i - 1):len(slice_path)]
                liver_batch = liver_path[batch_size * (i - 1):len(liver_path)]

                append_num = (batch_size * batch_num) - len(slice_path)
                for j in range(append_num):
                    slice_batch.append(slice_path[j])
                    liver_batch.append(liver_path[j])

        ######################################################################
        # slice_batch是一个batch，[[文件名],[文件名],[文件名]]
        batch_x = []
        for image_path in slice_batch:
            image_file = pydicom.dcmread(image_path)
            image_array = image_file.pixel_array
            image_array[image_array < -1024] = -1024
            image_array[image_array > 1024] = 1024
            image_array = (image_array + 1024) / 2048

            if crop == True:
                image_array = set_center_crop(x=image_array, width=width, height=height, center=center)
            batch_x.append(image_array)

        batch_y = []
        for image_path in liver_batch:
            image_file = pydicom.dcmread(image_path)
            image_array = image_file.pixel_array
            image_array[image_array == 255] = 1

            if crop == True:
                image_array = set_center_crop(x=image_array, width=width, height=height, center=center)
            batch_y.append(image_array)

        batch_x = np.asarray(batch_x)
        batch_x = batch_x.reshape((batch_x.shape[0], batch_x.shape[1], batch_x.shape[2], 1))
        batch_y = np.asarray(batch_y)
        batch_y = batch_y.reshape((batch_y.shape[0], batch_y.shape[1], batch_y.shape[2], 1))

        yield (batch_x, batch_y)


def shuffle_parallel_list(list_1, list_2):
    rand_num = random.randint(0, 100)
    random.seed(rand_num)
    random.shuffle(list_1)
    random.seed(rand_num)
    random.shuffle(list_2)
    return list_1, list_2

# base_dir = "F:/IRCAD/3Dircadb1/"
# slice_path_list, liver_path_list = get_slice_liver_path(base_dir)
# for i in get_batch(slice_path_list, liver_path_list, batch_size=4, crop=True, center=(150, 245), width=224, height=224):
#     batch_x = i[0]
#     batch_y = i[1]
#     print(batch_x.shape)
#     break
