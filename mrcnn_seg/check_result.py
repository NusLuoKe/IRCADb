#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/16 17:46
# @File    : check_result.py
# @Author  : NUS_LuoKe


import os
import sys

import numpy as np
import pydicom
import tensorflow as tf

import mrcnn.model as modellib
from mrcnn import liver_mrcnn
from mrcnn import utils

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Path to Livers trained weights
# LIVERS_MODEL_PATH = os.path.join(ROOT_DIR, "livers.h5")
LIVERS_MODEL_PATH = os.path.join(ROOT_DIR, "logs/liver20180716T1949/mask_rcnn_liver_0160.h5")

config = liver_mrcnn.LiverConfig()


# Override the training configurations
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

# Useful if you're training a model on the same
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/gpu:1"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
TEST_MODE = "inference"

# Build validation dataset
base_dir = "../3Dircadb1"
# train_patient_id_list = list(range(1, 19))
patient_id_list = [19]
dataset = liver_mrcnn.LiverDataset()
# filter_liver=False, all slice should make a prediction
dataset.load_livers(base_dir=base_dir, patient_id_list=patient_id_list, filter_liver=False)

# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

# Set weights file path
weights_path = LIVERS_MODEL_PATH

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

for image_id in dataset.image_ids:
    info = dataset.image_info[image_id]
    print("ori slice path %s" % info['path'])

    # Get mask directory from image path
    mask_name = os.path.split(info['path'])[1]
    mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "MASKS_DICOM/liver")

    # Read mask files
    mask_path = os.path.join(mask_dir, mask_name)
    mask_file = pydicom.read_file(mask_path)

    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id,
                                                                              use_mini_mask=False)
    info = dataset.image_info[image_id]
    print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                           dataset.image_reference(image_id)))
    # Run object detection
    results = model.detect([image], verbose=1)

    # generating masks
    # Get predictions of mask head
    mrcnn = model.run_graph([image], [
        ("detections", model.keras_model.get_layer("mrcnn_detection").output),
        ("masks", model.keras_model.get_layer("mrcnn_mask").output),
    ])

    # Get detection class IDs. Trim zero padding.
    det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
    det_count = np.where(det_class_ids == 0)[0][0]
    det_class_ids = det_class_ids[:det_count]

    print("{} detections: {}".format(
        det_count, np.array(dataset.class_names)[det_class_ids]))

    # Masks
    det_boxes = utils.denorm_boxes(mrcnn["detections"][0, :, :4], image.shape[:2])
    det_mask_specific = np.array([mrcnn["masks"][0, i, :, :, c]
                                  for i, c in enumerate(det_class_ids)])
    det_masks = np.array([utils.unmold_mask(m, det_boxes[i], image.shape)
                          for i, m in enumerate(det_mask_specific)])

    if len(det_masks) == 0:
        det_masks = np.zeros((1, 512, 512))
    elif det_masks.shape[0] != 1:
        # eg: det_masks = (2, 512, 512) to (1, 512, 512)
        det_masks = np.reshape(np.logical_or(det_masks[0, :, :], det_masks[1, :, :]), (1, 512, 512))

    img = det_masks.reshape(det_masks.shape[1], det_masks.shape[2])
    flip_img = np.flip(img, axis=0)

    mask_file.pixel_array.flat = np.int16(flip_img)
    mask_file.PixelData = mask_file.pixel_array.tostring()
    save_dir = "../prediction_results_p19"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    mask_file.save_as(os.path.join(save_dir, os.path.split(mask_path)[1]))
    print("mask path %s" % mask_path)
    print()
    print("@@@" * 30)
    print()
