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
from scipy.misc import imresize
from skimage.measure import regionprops


def set_square_crop(x, min_row, min_col, max_row, max_col):
    # return x[min_row:max_row, min_col:max_col]
    if max_row - min_row > max_col - min_col:
        return x[min_row:max_row, min_row:max_row]
    else:
        return x[min_col:max_col, min_col:max_col]


def set_center_crop(x, width, height, center, row_index=0, col_index=1):
    '''
    :param x:input, numpy array
    :param width: width of the output
    :param height: height of the output
    :param center: coordinate of the crop center
    :param row_index: row index
    :param col_index: column index
    :return: numpy array cropped image
    '''
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


def get_slice_liver_path(base_dir, patient_id_list=None, shuffle=True):
    slice_path_list = []  # 将所有图片的文件名的路径存成slice_path_list
    liver_path_list = []  # 将所有图片中liver的mask的文件名的路径存成liver_path_list
    # patient__num = len(os.listdir(base_dir))
    # patient_id_list = [i for i in range(1, 20)]  # 留出最后一个病人的数据做验证集

    for patient_id in patient_id_list:
        patient_dicom_path = "3Dircadb1." + str(patient_id) + "/PATIENT_DICOM"
        liver_dicom_path = "3Dircadb1." + str(patient_id) + "/MASKS_DICOM/portalvein"
        # liver_dicom_path = "3Dircadb1." + str(patient_id) + "/MASKS_DICOM/liver"
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


def filter_useless_data(slice_path_list, liver_path_list):
    x_with_vessel = []
    y_with_vessel = []
    vessel_num = 0
    for image_path in liver_path_list:
        image_file = pydicom.dcmread(image_path)
        image_array = image_file.pixel_array
        if np.sum(image_array) > 0:
            vessel_num += 1
            idx = liver_path_list.index(image_path)
            y_with_vessel.append(image_path)
            x_with_vessel.append(slice_path_list[idx])
    return x_with_vessel, y_with_vessel, vessel_num


def get_batch_crop_center(slice_path, liver_path, batch_size, crop_by_center=False, center=None, height=None,
                          width=None):
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
            image_array = (image_array + 1024.) / 2048.

            if crop_by_center:
                image_array = set_center_crop(x=image_array, width=width, height=height, center=center)
            batch_x.append(image_array)

        batch_y = []
        for image_path in liver_batch:
            image_file = pydicom.dcmread(image_path)
            image_array = image_file.pixel_array
            image_array[image_array == 255] = 1

            if crop_by_center:
                image_array = set_center_crop(x=image_array, width=width, height=height, center=center)
            batch_y.append(image_array)

        batch_x = np.asarray(batch_x)
        batch_x = batch_x.reshape((batch_x.shape[0], batch_x.shape[1], batch_x.shape[2], 1))
        batch_y = np.asarray(batch_y)
        batch_y = batch_y.reshape((batch_y.shape[0], batch_y.shape[1], batch_y.shape[2], 1))

        yield (batch_x, batch_y)


def enlarge_slice(batch_x, batch_y, batch_size, length=512, width=512):
    '''
    将一个batch的slice里面，选出mask最大的一张，以这张的有label的部分选出外接正方形，
    按照这个正方形的大小截取这个batch的所有图片.
    :param batch_x:
    :param batch_y:
    :param batch_size:
    :param length:
    :param width:
    :return:
    '''
    if np.sum(np.sum(batch_y, axis=(1, 2, 3))) != 0:
        # find the box to crop slices
        largest_mask_slice = batch_y[np.argmax(np.sum(batch_y, axis=(1, 2, 3)))]
        largest_mask_slice = np.reshape(largest_mask_slice, (largest_mask_slice.shape[0], largest_mask_slice.shape[1]))

        props = regionprops(largest_mask_slice)
        box = list(props[0].bbox)

        # apply the box to all slices in the batch
        cropped_batch_y = []
        for i in range(batch_size):
            slice = batch_y[i]
            slice = np.reshape(slice, (slice.shape[0], slice.shape[1]))
            slice = set_square_crop(slice, box[0], box[1], box[2], box[3])
            slice = imresize(slice, (length, width))
            cropped_batch_y.append(slice)

        cropped_batch_y = np.asarray(cropped_batch_y)
        cropped_batch_y = cropped_batch_y.reshape(
            (cropped_batch_y.shape[0], cropped_batch_y.shape[1], cropped_batch_y.shape[2], 1))

        cropped_batch_x = []
        for i in range(batch_size):
            slice = batch_x[i]
            slice = np.reshape(slice, (slice.shape[0], slice.shape[1]))
            slice = set_square_crop(slice, box[0], box[1], box[2], box[3])
            slice = imresize(slice, (length, width))
            cropped_batch_x.append(slice)

        cropped_batch_x = np.asarray(cropped_batch_x)
        cropped_batch_x = cropped_batch_x.reshape(
            (cropped_batch_x.shape[0], cropped_batch_x.shape[1], cropped_batch_x.shape[2], 1))
        return cropped_batch_x, cropped_batch_y
    else:
        return batch_x, batch_y


def shuffle_parallel_list(list_1, list_2):
    '''
    :param list_1:
    :param list_2:
    :return: shuffled list_1 and list_2 with same seed

    eg:
    :param list_1:[1,2,3,4,5]
    :param list_2:[1,2,3,4,5]
    :return list_1:[2,3,5,4,1]
             list_2:[2,3,5,4,1]
    '''

    rand_num = random.randint(0, 100)
    random.seed(rand_num)
    random.shuffle(list_1)
    random.seed(rand_num)
    random.shuffle(list_2)
    return list_1, list_2
