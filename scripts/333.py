#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/2 16:16
# @File    : 333.py
# @Author  : NUS_LuoKe

# from tf_dicom.load_dicom import *
# import matplotlib.pyplot as plt
# from keras.preprocessing.image import ImageDataGenerator
#
# base_dir = "F:/IRCAD/3Dircadb1/"
# train_patient_id_list = list(range(1, 5))
# train_slice_path_list, train_liver_path_list = get_slice_liver_path(base_dir, patient_id_list=train_patient_id_list,
#                                                                     shuffle=True)
#
# for train_batch_x_y in get_batch(train_slice_path_list, train_liver_path_list, batch_size=4, crop_by_center=False):
#     train_batch_x = train_batch_x_y[0]
#     train_batch_y = train_batch_x_y[1]
#
#     data_gen_args = dict(rotation_range=90)
#     img_datagen = ImageDataGenerator(**data_gen_args)
#     mask_datagen = ImageDataGenerator(**data_gen_args)
#
#     seed = 1
#     img_datagen.fit(train_batch_x, augment=True, seed=seed)
#     mask_datagen.fit(train_batch_y, augment=True, seed=seed)
#
#     img_gen = img_datagen.flow(train_batch_x, batch_size=4, seed=seed)
#     mask_gen = mask_datagen.flow(train_batch_y, batch_size=4, seed=seed)
#
#     print(len(img_gen[0]))
#     print(img_gen[0].shape)
#
#     one_img = img_gen[0][0].reshape(img_gen[0].shape[1], img_gen[0].shape[2])
#     one_mask = mask_gen[0][0].reshape(mask_gen[0].shape[1], mask_gen[0].shape[2])
#     print(one_img.shape)
#
#     plt.imshow(one_img, cmap=plt.cm.gray)
#     plt.show()
#
#     plt.imshow(one_mask, cmap=plt.cm.gray)
#     plt.show()
#
#     break
