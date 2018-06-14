#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/12 14:35
# @File    : load_dicom.py
# @Author  : NUS_LuoKe

import os
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import pydicom


def get_slice_liver_path(base_dir, shuffle=True):
    slice_path_list = []  # 将所有图片的文件名的路径存成slice_path_list
    liver_path_list = []  # 将所有图片中liver的mask的文件名的路径存成liver_path_list
    patient__num = len(os.listdir(base_dir))
    patient_id_list = [i for i in range(1, patient__num + 1)]

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


def get_tra_val_test_set(slice_path, liver_path, tra_ratio=0.8, val_ratio=0.1):
    '''
    :param slice_path: 存放所有slice文件路径的list
    :param liver_path: 存放所有slice对应的liver文件路径的list
    :param tra_ratio: 训练集的比例
    :param val_ratio: 验证集的比例
    :return: list type. eg: training_set = [[1], [2]], [1]和[2]为存放img和label的文件路径的list
    '''
    item_num = len(slice_path)
    train_num = int(item_num * tra_ratio)
    val_num = int(item_num * val_ratio)

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
            batch_x.append(image_array)

        batch_y = []
        for image_path in liver_batch:
            image_file = pydicom.dcmread(image_path)
            image_array = image_file.pixel_array
            batch_y.append(image_array)

        batch_x = np.asarray(batch_x)
        batch_x = batch_x.reshape((batch_x.shape[0], batch_x.shape[1], batch_x.shape[2], 1))
        batch_y = np.asarray(batch_y)
        batch_y = batch_y.reshape((batch_y.shape[0], batch_y.shape[1], batch_y.shape[2], 1))

        yield (batch_x, batch_y)


base_dir = "F:/IRCAD/3Dircadb1/"
slice_path_list, liver_path_list = get_slice_liver_path(base_dir, shuffle=True)
for i in get_batch(slice_path=slice_path_list, liver_path=liver_path_list, batch_size=4):
    batch_x = i[0]

    print(batch_x.shape)
    print()
    break


'''
# 这个方法生成batch在训练的时候会报错，OOM
def get_batch(slice_path, liver_path, batch_size):    
    batch_num = int(ceil(len(slice_path) / batch_size))

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
            image = sitk.ReadImage(image_path)
            image_array = sitk.GetArrayFromImage(image)  # z, y, x, It is a numpy.ndarray
            # normalization

            image_array_shape = image_array.shape
            image_array = image_array.reshape(
                (image_array_shape[2], image_array_shape[1], image_array_shape[0]))  # reshape为x, y, z
            batch_x.append(image_array)  # batch_x = [[array],[array],[array],[array]]
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


base_dir = "F:/IRCAD/3Dircadb1/"
slice_path_list, liver_path_list = get_slice_liver_path(base_dir, shuffle=True)
for i in get_batch(slice_path=slice_path_list, liver_path=liver_path_list, batch_size=4):
    batch_x = i[0]
    batch_y = i[1]

    print(batch_x.shape)
    print(batch_y.shape)
    print()
'''


# def read_file_queue(slice_path, liver_path, q_num):
#     '''
#     将所有的slice的文件名分成q_num个文件名list，在后续方法中，加载每个list，将list中的文件读取出来，再生成batch
#     a = read_file_queue[], a = [[file_names],[file_names],...,[file_names]]
#     :param slice_path:
#     :param liver_path:
#     :param q_num:
#     :return:
#     '''
#     q_length = int(math.ceil(len(slice_path) / q_num))  # 每个queue中有多少个文件
#
#     slice_file_name_batch_list = []
#     liver_file_name_batch_list = []
#     for i in range(1, q_num + 1):
#         if (q_num * q_length) == len(slice_path):
#             slice_batch = slice_path[q_length * (i - 1):q_length * i]
#             liver_batch = liver_path[q_length * (i - 1):q_length * i]
#
#         elif (q_num * q_length) != len(slice_path):
#             if i < q_length:
#                 slice_batch = slice_path[q_length * (i - 1):q_length * i]
#                 liver_batch = liver_path[q_length * (i - 1):q_length * i]
#             else:
#                 slice_batch = slice_path[q_length * (i - 1):len(slice_path)]
#                 liver_batch = liver_path[q_length * (i - 1):len(liver_path)]
#
#                 append_num = (q_num * q_length) - len(slice_path)
#                 for j in range(append_num):
#                     # slice_batch是一个batch，[[文件名],[文件名],[文件名]]
#                     slice_batch.append(slice_path[j])
#                     liver_batch.append(liver_path[j])
#         slice_file_name_batch_list.append(slice_batch)
#         liver_file_name_batch_list.append(liver_batch)
#     return slice_file_name_batch_list, liver_file_name_batch_list
#
#
# def chunks(l, n):
#     '''yield successive n-sized chunks from l'''
#     for i in range(0, len(l), n):
#         yield l[i:i + n]
#
#
# def get_batch(image_list, chunk_size, visualize_one=False):
#     image_array_list = []
#     for image_path in image_list:
#         image_file = pydicom.dcmread(image_path)
#         image_array = image_file.pixel_array
#         image_array_list.append(image_array)
#
#     if visualize_one:
#         plt.imshow(image_array_list[0])
#         plt.show()
#
#     new_images = []
#     for image_chunk in chunks(image_array_list, chunk_size):
#         # image_chunk = list(map(mean, zip(*image_chunk)))
#         new_images.append(image_chunk)
#
#     return new_images
#
#
# base_dir = "F:/IRCAD/3Dircadb1/"
# slice_path_list, liver_path_list = get_slice_liver_path(base_dir, shuffle=True)
# slice_file_name_batch_list, liver_file_name_batch_list = read_file_queue(slice_path_list, liver_path_list, 20)
# # print(len(slice_file_name_batch_list))
# print(slice_file_name_batch_list[0])
# # print(liver_file_name_batch_list[0])
# # print(slice_file_name_batch_list[1])
# # print(liver_file_name_batch_list[1])
# slice_array = get_batch(slice_file_name_batch_list[0], chunk_size=4)
# print(len(slice_array[0]))  # slice_array[0]是一个batch
# print(slice_array[0][0].shape)
# a = slice_array[0][0].reshape(512, 512, 1)
# plt.imshow(a)
# plt.show()


