#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/12 17:10
# @File    : load_dicom.py
# @Author  : NUS_LuoKe

import os
import random

import numpy as np
import pydicom


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
            if os.path.basename(single_slice_path)[0:5] == "image":
                slice_path_list.append(single_slice_path)

        for mask in os.listdir(mask_path):
            single_mask_path = os.path.join(mask_path, mask)
            if os.path.basename(single_mask_path)[0:5] == "image":
                mask_path_list.append(single_mask_path)

        if shuffle == True:
            rand_num = random.randint(0, 100)
            random.seed(rand_num)
            random.shuffle(slice_path_list)
            random.seed(rand_num)
            random.shuffle(mask_path_list)

    return slice_path_list, mask_path_list


def filter_useless_data(slice_path_list, mask_path_list, reserve_some=False, reserve_num=None):
    '''
     filter out those slices with no mask.
     (eg: filter out those slices with no vessel in 3Dircadb1 data set)

     :param slice_path_list: the list contains file path of all slices
     :param mask_path_list: the list contains file path of all masks
     :param reserve_some: default to be False, when True, can reserve some slices without mask(liver, etc) in the list
     :param reserve_num: number of reserved slices without mask
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
        else:
            count_reserve = 0
            while reserve_some:
                mask_num += 1
                count_reserve += 1
                idx = mask_path_list.index(image_path)
                y_with_mask.append(image_path)
                x_with_mask.append(slice_path_list[idx])
                if count_reserve == reserve_num:
                    reserve_some = False

    return x_with_mask, y_with_mask, mask_num


# base_dir = "F:/IRCAD/3Dircadb1/"
# train_patient_id_list = [1]
# train_slice_path_list, train_mask_path_list = get_slice_mask_path(base_dir, patient_id_list=train_patient_id_list,
#                                                                   shuffle=True)
# train_x_with_vessel, train_y_with_vessel, train_vessel_num = filter_useless_data(train_slice_path_list,
#                                                                                  train_mask_path_list,
#                                                                                  reserve_some=True, reserve_num=10)
# print(train_vessel_num)
