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
    '''
    Input params are coordinates of a rectangle box,
    and the purpose is to find the proper external square to crop the input image x.

    :param x: numpy array, an image
    :param min_row: min_row of the box
    :param min_col: min_col of the box
    :param max_row: max_row of the box
    :param max_col: max_col of the box
    :return: numpy array, cropped x
    '''
    if max_row - min_row > max_col - min_col:
        gap = max_row - min_row
        col = min_col if min_col + gap < x.shape[1] else x.shape[1] - gap
        return x[min_row:max_row, col:col + gap]
    else:
        gap = max_col - min_col
        row = min_row if min_row + gap < x.shape[0] else x.shape[0] - gap
        return x[row:row + gap, min_col:max_col]


def set_center_crop(x, width, height, center, row_index=0, col_index=1):
    '''
    Set a center to crop image

    :param x:numpy array, an image
    :param width: width of the output
    :param height: height of the output
    :param center: coordinate of the crop center
    :param row_index: row index
    :param col_index: column index
    :return: numpy array, cropped x
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


def get_slice_mask_path(base_dir, patient_id_list=None, shuffle=True):
    '''
    save file names of slices and masks in two list

    :param base_dir: directory of the 3Dircadb1 data set
    :param patient_id_list: a list of patients' id
    :param shuffle: if shuffle, the slices of all  patients in patient_id_list will be shuffled
    :return:
    slice_path_list: a list contains file path of slices
    mask_path_list: a list contains file path of masks
    '''
    slice_path_list = []
    mask_path_list = []

    for patient_id in patient_id_list:
        patient_dicom_path = "3Dircadb1." + str(patient_id) + "/PATIENT_DICOM"
        mask_dicom_path = "3Dircadb1." + str(patient_id) + "/MASKS_DICOM/portalvein"
        slice_path = os.path.join(base_dir, patient_dicom_path)
        mask_path = os.path.join(base_dir, mask_dicom_path)

        for slice in os.listdir(slice_path):
            single_slice_path = os.path.join(slice_path, slice)
            slice_path_list.append(single_slice_path)

        for liver in os.listdir(mask_path):
            single_liver_path = os.path.join(mask_path, liver)
            mask_path_list.append(single_liver_path)

        if shuffle == True:
            rand_num = random.randint(0, 100)
            random.seed(rand_num)
            random.shuffle(slice_path_list)
            random.seed(rand_num)
            random.shuffle(mask_path_list)

    return slice_path_list, mask_path_list


def filter_useless_data(slice_path_list, mask_path_list):
    '''
    filter out those slices with no mask.
    (eg: filter out those slices with no vessel in 3Dircadb1 data set)

    :param slice_path_list: the list contains file path of all slices
    :param mask_path_list: the list contains file path of all masks
    :return:
    x_with_mask: list, contains file path of slices with mask
    y_with_mask: list, contains file path of masks for those slices with mask
    mask_num: the total number of slices with mask
    '''
    x_with_mask = []
    y_with_mask = []
    mask_num = 0
    for image_path in mask_path_list:
        image_file = pydicom.dcmread(image_path)
        image_array = image_file.pixel_array
        if np.sum(image_array) > 0:
            mask_num += 1
            idx = mask_path_list.index(image_path)
            y_with_mask.append(image_path)
            x_with_mask.append(slice_path_list[idx])
    return x_with_mask, y_with_mask, mask_num


def get_batch(slice_path, mask_path, batch_size, crop_by_center=False, center=None, height=None,
              width=None):
    '''
    generator, generating batches for neural networks.

    :param slice_path: a list contains the file path of slices
    :param mask_path: a list contains the file path of masks
    :param batch_size: batch size
    :param crop_by_center: crop the image by center
    :param center: param for the crop_by_center method
    :param height: param for the crop_by_center method
    :param width: param for the crop_by_center method
    :yield:
    batch_x: a batch of slices
    batch_y: a batch of corresponding masks
    '''
    batch_num = int(math.ceil(len(slice_path) / batch_size))

    for i in range(1, batch_num):
        if (batch_size * batch_num) == len(slice_path):
            slice_batch = slice_path[batch_size * (i - 1):batch_size * i]
            liver_batch = mask_path[batch_size * (i - 1):batch_size * i]

        elif (batch_size * batch_num) != len(slice_path):
            if i < batch_num:
                slice_batch = slice_path[batch_size * (i - 1):batch_size * i]
                liver_batch = mask_path[batch_size * (i - 1):batch_size * i]
            else:
                slice_batch = slice_path[batch_size * (i - 1):len(slice_path)]
                liver_batch = mask_path[batch_size * (i - 1):len(mask_path)]

                append_num = (batch_size * batch_num) - len(slice_path)
                for j in range(append_num):
                    slice_batch.append(slice_path[j])
                    liver_batch.append(mask_path[j])

        # slice_batch is a batch in form of [[files' path],[files' path],[files' path]]
        batch_x = []
        for image_path in slice_batch:
            image_file = pydicom.dcmread(image_path)
            image_array = image_file.pixel_array
            # data truncation
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
            # data truncation
            image_array[image_array > 0] = 1

            if crop_by_center:
                image_array = set_center_crop(x=image_array, width=width, height=height, center=center)
            batch_y.append(image_array)

        batch_x = np.asarray(batch_x)
        batch_x = batch_x.reshape((batch_x.shape[0], batch_x.shape[1], batch_x.shape[2], 1))
        batch_y = np.asarray(batch_y)
        batch_y = batch_y.reshape((batch_y.shape[0], batch_y.shape[1], batch_y.shape[2], 1))

        yield (batch_x, batch_y)


def resize_batch(batch_x, batch_y, batch_size, length=512, width=512):
    '''
    select the slice with largest mask in the batch,
    then select the circumscribed square with the label part of this slice.
    the size of this block crop all the images in this batch.

    :param batch_x: numpy array. a batch of slice
    :param batch_y: numpy array. a batch of mask
    :param batch_size: batch size
    :param length: length of the output image
    :param width: width of the output image
    :return:
     batch_x: a resized batch of slice
     batch_y: a resized batch of mask
    '''
    if np.sum(np.sum(batch_y, axis=(1, 2, 3))) != 0:
        # find the box to crop slices
        largest_mask_slice = batch_y[np.argmax(np.sum(batch_y, axis=(1, 2, 3)))]
        largest_mask_slice = np.reshape(largest_mask_slice, (largest_mask_slice.shape[0], largest_mask_slice.shape[1]))

        props = regionprops(largest_mask_slice)
        # Bounding box ``(min_row, min_col, max_row, max_col)
        box = props[0].bbox

        # apply the box to all slices in the batch
        cropped_batch_y = []
        for i in range(batch_size):
            slice = batch_y[i]
            slice = np.reshape(slice, (slice.shape[0], slice.shape[1]))
            slice = set_square_crop(slice, min_row=box[0], min_col=box[1], max_row=box[2], max_col=box[3])
            slice = imresize(slice, (length, width))
            cropped_batch_y.append(slice)

        cropped_batch_y = np.asarray(cropped_batch_y)
        cropped_batch_y = cropped_batch_y.reshape(
            (cropped_batch_y.shape[0], cropped_batch_y.shape[1], cropped_batch_y.shape[2], 1))

        cropped_batch_x = []
        for i in range(batch_size):
            slice = batch_x[i]
            slice = np.reshape(slice, (slice.shape[0], slice.shape[1]))
            slice = set_square_crop(slice, min_row=box[0], min_col=box[1], max_row=box[2], max_col=box[3])
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
    shuffle two parallel list

    :param list_1: list 1
    :param list_2: list 2
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
