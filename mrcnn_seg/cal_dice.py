#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/18 9:29
# @File    : cal_dice.py
# @Author  : NUS_LuoKe


import tensorflow as tf
import os
import pydicom
import numpy as np


def dice_hard_coe(output, target, threshold=0.5, axis=(1, 2, 3), smooth=1e-5):
    output = tf.cast(output > threshold, dtype=tf.float32)
    target = tf.cast(target > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(output, target), axis=axis)
    l = tf.reduce_sum(output, axis=axis)
    r = tf.reduce_sum(target, axis=axis)
    hard_dice = (2. * inse + smooth) / (l + r + smooth)
    hard_dice = tf.reduce_mean(hard_dice)
    return hard_dice


def get_pred_mask_path(pred_dir, mask_dir):
    preds_list = []
    masks_list = []
    for pred in os.listdir(pred_dir):
        pred_path = os.path.join(pred_dir, pred)
        preds_list.append(pred_path)

    for mask in os.listdir(mask_dir):
        mask_path = os.path.join(mask_dir, mask)
        masks_list.append(mask_path)

    return preds_list, masks_list


if __name__ == "__main__":
    # for patient 1
    pred_dir = "D:/MRCNN/v2/mrcnn_prediction_p1"
    mask_dir = "F:/IRCAD/3Dircadb1/3Dircadb1.1/MASKS_DICOM/liver"
    preds_list, masks_list = get_pred_mask_path(pred_dir, mask_dir)
    print(preds_list)
    print(masks_list)

    for i in range(len(preds_list)):
        pred_path = preds_list[i]
        mask_path = masks_list[i]
        slice_id = os.path.split(pred_path)[-1].split("_")[-1]

        pred_file = pydicom.read_file(pred_path)
        mask_file = pydicom.read_file(mask_path)
        pred_arr = pred_file.pixel_array
        mask_arr = mask_file.pixel_array
        mask_flip = np.flip(mask_arr, axis=0)

        pred = pred_arr.reshape(1, 512, 512, 1)
        mask = mask_flip.reshape(1, 512, 512, 1)

        hard_dice = dice_hard_coe(mask, pred, threshold=0.5, axis=(1, 2, 3), smooth=1e-5)
        with tf.Session() as sess:
            dice = sess.run(hard_dice)
            print("slice id: %s" % slice_id + "  dice: %s " % dice)
