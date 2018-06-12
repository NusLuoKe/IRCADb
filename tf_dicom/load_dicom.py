#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/12 14:35
# @File    : load_dicom.py
# @Author  : NUS_LuoKe

import os
import numpy as np
from math import ceil
import SimpleITK as sitk
import random
import matplotlib.pyplot as plt

base_dir = "F:/IRCAD/3Dircadb1/"


def get_slice_liver_path(base_dir, shuffle=True):
    slice_path_list = []  # 将所有图片的文件名的路径存成slice_path_list
    liver_path_list = []  # 将所有图片中liver的mask的文件名的路径存成liver_path_list
    patient__num = len(os.listdir(base_dir))
    patient_id_list = [i for i in range(1, patient__num + 1)]

    for patient_id in patient_id_list:
        patient_dicom_path = "3Dircadb1." + str(patient_id) + "/PATIENT_DICOM"
        liver_dicom_path = "3Dircadb1." + str(patient_id) + "/MASKS_DICOM/liver"
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


def get_tra_val_test_set(slice_path, liver_path, tra_ratio=0.8, val_ratio=0.1):
    '''
    :param slice_path: 存放所有slice文件路径的list
    :param liver_path: 存放所有slice对应的liver文件路径的list
    :param tra_ratio: 训练集的比例
    :param val_ratio: 验证集的比例
    :return: 返回的为参数类型为list. eg: training_set = [[1], [2]], [1]和[2]为存放img和label的文件路径的list
    '''
    item_num = len(slice_path)
    train_num = item_num * tra_ratio
    val_num = item_num * val_ratio

    training_set_slice = slice_path[:train_num]
    training_set_liver = liver_path[:train_num]
    validation_set_slice = slice_path[train_num:train_num + val_num]
    validation_set_liver = liver_path[train_num:train_num + val_num]
    test_set_slice = slice_path[train_num + val_num:]
    test_set_liver = liver_path[:train_num]

    training_set = [training_set_slice, training_set_liver]
    validation_set = [validation_set_slice, validation_set_liver]
    test_set = [test_set_slice, test_set_liver]
    return training_set, validation_set, test_set


def get_batch(slice_path, liver_path, batch_size):
    batch_num = int(ceil(len(slice_path) / batch_size))

    for i in range(1, batch_num + 1):
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

        batch_x = []
        for image_path in slice_batch:
            image = sitk.ReadImage(image_path)
            image_array = sitk.GetArrayFromImage(image)  # z, y, x, It is a numpy.ndarray
            # normalization

            image_array_shape = image_array.shape
            image_array = image_array.reshape(
                (image_array_shape[2], image_array_shape[1], image_array_shape[0]))  # reshape为x, y, z
            batch_x.append(image_array)
        batch_x = np.asarray(batch_x)

        batch_y = []
        for image_path in liver_batch:
            image = sitk.ReadImage(image_path)
            image_array = sitk.GetArrayFromImage(image)  # z, y, x, It is a numpy.ndarray
            # normalization

            image_array_shape = image_array.shape
            image_array = image_array.reshape(
                (image_array_shape[2], image_array_shape[1], image_array_shape[0]))  # reshape为x, y, z
            batch_y.append(image_array)
        batch_y = np.asarray(batch_y)

        yield (batch_x, batch_y)


'''
slice_path_list, liver_path_list = get_slice_liver_path(base_dir, shuffle=True)
for i in get_batch(slice_path_list, liver_path_list, 4):
    # print(i)
    # print(np.max(i[0]))
    # print(np.min(i[0]))
    # print("@" * 10)
    # print(np.max(i[1]))
    # print(np.min(i[1]))

    # print(i[0])
    print(i[0].shape)
    print(i[0][0].shape)  # i[0][0]为读取出来的单张dicom图片,shape = (512, 512, 1)
    # print(i[0][0][0][0][0].type)
    break

# aa_path = "F:/IRCAD/3Dircadb1/3Dircadb1.1/PATIENT_DICOM/image_0"
# img = sitk.ReadImage(aa_path)
# scalarImage = sitk.Cast(img, sitk.sitkUInt32)
# sitk.Show(scalarImage)
'''
