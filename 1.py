#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/21 9:39
# @File    : 1.py
# @Author  : NUS_LuoKe

'''
改series number
from __future__ import print_function

import pydicom

# series_number = 1
# for folder in ["D:/livertumor01/SERIES_0", "D:/livertumor01/SERIES_1", "D:/livertumor01/SERIES_2"]:
#     instance = 0
#     for file_name in [folder + '/image_{}.dcm'.format(i) for i in range(129)]:
#         print(file_name)
#         ds = pydicom.read_file(file_name)
#         ds.SeriesNumber = series_number
#         ds.InstanceNumber = instance
#         print(ds.SeriesNumber, ds.InstanceNumber)
#         ds.save_as(file_name)
#         instance += 1
#     series_number += 1


file_name = "D:/WANGGUOPING.CT.ABDOMEN_HX_CHEST_KI_20180309_170845187_002312.dcm"
ds = pydicom.read_file(file_name)

print(ds.BodyPartExamined)
# print()
# print(ds)
'''

'''
画图
import pydicom
import matplotlib.pyplot as plt

image_path = "F:/IRCAD/3Dircadb1/3Dircadb1.1/PATIENT_DICOM/image_0"
image_file = pydicom.dcmread(image_path)
image_array = image_file.pixel_array

plt.hist(image_array.flatten(), bins=80, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()

# Show some slice in the middle
plt.imshow(image_array)
plt.show()
'''

# 根据标签的范围裁剪图片
import matplotlib.pyplot as plt
from tf_dicom.load_dicom import *


def set_square_crop(x, min_row, min_col, max_row, max_col):
    # return x[min_row:max_row, min_col:max_col]
    if max_row - min_row > max_col - min_col:
        return x[min_row:max_row, min_row:max_row]
    else:
        return x[min_col:max_col, min_col:max_col]


length = 512
width = 512
base_dir = "F:/IRCAD/3Dircadb1/"
batch_size = 4
slice_path_list, liver_path_list = get_slice_liver_path(base_dir, patient_id_list=[1, 2])
for i in get_batch_crop_center(slice_path_list, liver_path_list, batch_size=batch_size, crop_by_center=False):
    batch_x = i[0]
    batch_y = i[1]

    if np.sum(np.sum(batch_y, axis=(1, 2, 3))) != 0:
        # find the box to crop slices
        largest_mask_slice = batch_y[np.argmax(np.sum(batch_y, axis=(1, 2, 3)))]
        largest_mask_slice = np.reshape(largest_mask_slice, (largest_mask_slice.shape[0], largest_mask_slice.shape[1]))
        plt.imshow(largest_mask_slice)
        plt.show()

        props = regionprops(largest_mask_slice)
        box = list(props[0].bbox)
        largest_mask_slice = set_square_crop(largest_mask_slice, box[0], box[1], box[2], box[3])
        plt.imshow(largest_mask_slice)
        plt.show()

        resize_lar = imresize(largest_mask_slice, (512, 512))
        plt.imshow(resize_lar)
        plt.show()

        # apply the box to all slices in the batch
        cropped_batch_y = []
        for i in range(batch_size):
            slice = batch_y[i]
            slice = np.reshape(slice, (slice.shape[0], slice.shape[1]))
            slice = set_square_crop(slice, box[0], box[1], box[2], box[3])
            slice = imresize(slice, (length, width))
            plt.imshow(slice)
            plt.show()
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
            plt.imshow(slice)
            plt.show()
            cropped_batch_x.append(slice)

        cropped_batch_x = np.asarray(cropped_batch_x)
        cropped_batch_x = cropped_batch_x.reshape(
            (cropped_batch_x.shape[0], cropped_batch_x.shape[1], cropped_batch_x.shape[2], 1))

        batch_x = cropped_batch_x
        batch_y = cropped_batch_y

    else:
        # if the mask of every slice in the batch is empty, then do not processing the original image
        pass

    break
